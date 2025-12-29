#include <iostream>
#include <fstream>
#include <cmath>
#include <complex>
#include <armadillo>
#include <boost/numeric/odeint.hpp>

using namespace std;
using namespace boost::numeric::odeint;

// System parameters
struct Params {
    // Polariton parameters
    double gamma1 = 1.0;      // Site 1 dissipation
    double gamma2 = 1.0;      // Site 2 dissipation
    double U1 = 0.0;          // Site 1 nonlinearity
    double U2 = 0.0;          // Site 2 nonlinearity
    
    // Coupling parameters
    double J = 0.0;           // Constant hopping
    double g = 1.0;           // Phonon coupling strength
    double delta = 0.0;       // Detuning
    
    // Phonon parameters
    double Omega = 20.0;      // Phonon frequency
    double Gamma = 0.05;      // Phonon damping
    
    // Reservoir parameters (site 1 only)
    double xi = 1.0;          // Reservoir coupling to site 1
    double tau = 600.0;       // Reservoir relaxation rate
    double P = 7.0;           // Pump power
    double alpha = sqrt(3.25); // Saturation parameter
};

Params params;

// State vector: [Re(phi1), Im(phi1), Re(phi2), Im(phi2), x, v, n]
typedef vector<double> state_type;

// Right-hand side of equations
void rhs(const state_type& y, state_type& dydt, double t) {
    // Resize output if needed
    if(dydt.size() != 7) dydt.resize(7);
    
    // Extract state variables
    complex<double> phi1(y[0], y[1]);
    complex<double> phi2(y[2], y[3]);
    double x = y[4];
    double v = y[5];
    double n = y[6];
    
    // Shorthand
    complex<double> I(0.0, 1.0);
    double gamma1 = params.gamma1;
    double gamma2 = params.gamma2;
    double U1 = params.U1;
    double U2 = params.U2;
    double J = params.J;
    double g = params.g;
    double delta = params.delta;
    double Omega = params.Omega;
    double Gamma = params.Gamma;
    double xi = params.xi;
    double tau = params.tau;
    double P = params.P;
    double alpha = params.alpha;
    
    // Compute phi1 derivative
    // dφ₁/dt = -i·φ₁·[-i·γ₁ + U₁·|φ₁|² + i·ξ·n] - i·(J + g·x)·exp(+i·Ω·t)·φ₂
    complex<double> H1 = phi1 * (-I * gamma1 + U1 * abs(phi1) * abs(phi1) + I * xi * n);
    complex<double> coupling_1to2 = (J + g * x) * polar(1.0, Omega * t) * phi2;
    complex<double> dphi1dt = -I * (H1 + coupling_1to2);
    
    // Compute phi2 derivative
    // dφ₂/dt = -i·φ₂·[-i·γ₂ + U₂·|φ₂|²] - i·(J + g·x)·exp(-i·Ω·t)·φ₁
    complex<double> H2 = phi2 * (-I * gamma2 + U2 * abs(phi2) * abs(phi2));
    complex<double> coupling_2to1 = (J + g * x) * polar(1.0, -Omega * t) * phi1;
    complex<double> dphi2dt = -I * (H2 + coupling_2to1);
    
    // Compute phonon derivatives
    // dx/dt = v
    double dxdt = v;
    
    // dv/dt = -Ω²·x - Γ·v + 2·Ω·Γ·Re[g·φ₁·φ₂*·exp(-i·Ω·t)]
    complex<double> backaction = g * phi1 * conj(phi2) * polar(1.0, -Omega * t);
    double dvdt = -Omega * Omega * x - Gamma * v - 2.0 * Omega * Gamma * real(backaction);
    
    // Compute reservoir derivative
    // dn/dt = τ·[P - n·(1 + α²·|φ₁|²)]
    double intensity1 = abs(phi1) * abs(phi1);
    double dndt = tau * (P - n * (1.0 + alpha * alpha * intensity1));
    
    // Pack derivatives
    dydt[0] = real(dphi1dt);
    dydt[1] = imag(dphi1dt);
    dydt[2] = real(dphi2dt);
    dydt[3] = imag(dphi2dt);
    dydt[4] = dxdt;
    dydt[5] = dvdt;
    dydt[6] = dndt;
}

int main(int argc, char* argv[]) {
    if(argc != 4){
        cout << "Usage: ./program <steps> <index> <output_file>" << endl;
        return 1;
    }
    
    size_t steps = stoi(argv[1]);
    size_t job_index = stoi(argv[2]);
    auto file_name = argv[3];
    
    double driving_start = 0.0;
    double driving_stop = 14.0;
    double driving_value = driving_start + (driving_stop - driving_start) * job_index / (steps - 1.0);
    
    params.P = driving_value;
    
    // Random initial conditions
    srand(time(0) + job_index);
    auto uniform = []() { return (double)rand() / RAND_MAX; };
    
    // Initial state
    state_type y(7);
    y[0] = uniform();           // Re(phi1)
    y[1] = uniform();           // Im(phi1)
    y[2] = 10.0 * uniform();    // Re(phi2)
    y[3] = 10.0 * uniform();    // Im(phi2)
    y[4] = 10.0 * uniform();    // x
    y[5] = 200.0 * uniform();   // v
    y[6] = 0.1 + 0.5 * uniform(); // n
    
    cout << "=== INITIAL STATE ===" << endl;
    cout << "Driving power: " << driving_value << endl;
    cout << "phi1 = " << y[0] << " + " << y[1] << "i" << endl;
    cout << "phi2 = " << y[2] << " + " << y[3] << "i" << endl;
    cout << "x = " << y[4] << ", v = " << y[5] << endl;
    cout << "n = " << y[6] << endl;
    
    // Integrator
    typedef runge_kutta_dopri5<state_type> error_stepper_type;
    typedef controlled_runge_kutta<error_stepper_type> controlled_stepper_type;
    controlled_stepper_type stepper;
    
    // Transient 
    double t = 0.0;
    double dt = 1e-3;
    size_t transient = 5e6;
    
    for(size_t i = 0; i < transient; ++i) {
        stepper.try_step(rhs, y, t, dt);
        
        if(i == 1000) {
            cout << "\n=== AFTER 1000 STEPS ===" << endl;
            cout << "Time: " << t << endl;
            cout << "|phi1|² = " << (y[0]*y[0] + y[1]*y[1]) << endl;
            cout << "|phi2|² = " << (y[2]*y[2] + y[3]*y[3]) << endl;
            cout << "x = " << y[4] << ", v = " << y[5] << endl;
            cout << "n = " << y[6] << endl;
            
            complex<double> phi1(y[0], y[1]);
            complex<double> phi2(y[2], y[3]);
            complex<double> backaction = phi1 * conj(phi2) * polar(1.0, -params.Omega * t);
            cout << "phi1*phi2* = " << (phi1 * conj(phi2)) << endl;
            cout << "Backaction force = " << (2.0 * params.Omega * params.Gamma * real(backaction)) << endl;
        }
    }
    
    cout << "\n=== AFTER FULL TRANSIENT ===" << endl;
    cout << "Time: " << t << endl;
    cout << "|phi1|² = " << (y[0]*y[0] + y[1]*y[1]) << endl;
    cout << "|phi2|² = " << (y[2]*y[2] + y[3]*y[3]) << endl;
    cout << "x = " << y[4] << endl;
    cout << "n = " << y[6] << endl;
    
    // Stationary averaging
    size_t stationary = 1e5;
    double avg_phi1 = 0.0;
    double avg_phi2 = 0.0;
    double avg_x2 = 0.0;
    double avg_n = 0.0;
    
    for(size_t i = 0; i < stationary; ++i) {
        stepper.try_step(rhs, y, t, dt);
        avg_phi1 += y[0]*y[0] + y[1]*y[1];
        avg_phi2 += y[2]*y[2] + y[3]*y[3];
        avg_x2 += y[4]*y[4];
        avg_n += y[6];
    }
    
    // Write output
    ofstream out(file_name);
    out << driving_value << "\t" 
        << avg_phi1/stationary << "\t"
        << avg_phi2/stationary << "\t"
        << avg_x2/stationary << "\t"
        << avg_n/stationary << endl;
    out.close();
    
    cout << "\n=== AVERAGES ===" << endl;
    cout << "<|phi1|²> = " << avg_phi1/stationary << endl;
    cout << "<|phi2|²> = " << avg_phi2/stationary << endl;
    cout << "<x²> = " << avg_x2/stationary << endl;
    cout << "<n> = " << avg_n/stationary << endl;
    
    return 0;
}