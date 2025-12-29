#pragma once
#include <fstream>
#include <sstream>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <armadillo>
#include <boost/numeric/odeint.hpp>
constexpr arma::cx_double I(0.0, 1.0);
using stationary_stepper = boost::numeric::odeint::runge_kutta4<arma::vec>;
using error_stepper = boost::numeric::odeint::runge_kutta_dopri5<arma::vec>;
using adaptive_stepper = boost::numeric::odeint::controlled_runge_kutta<error_stepper>;
class PolaritonMode;
class PhononMode;
class NonResonantDriving;

// namespace redefinitions for the compatibility of armadillo with odeint.
namespace boost { namespace numeric { namespace odeint {

// ===================== arma::vec =====================
template <>
struct is_resizeable<arma::vec> : boost::true_type {};

template <>
struct same_size_impl<arma::vec, arma::vec>
{
    static bool same_size(const arma::vec& x, const arma::vec& y)
    {
        return x.n_elem == y.n_elem;
    }
};

template <>
struct resize_impl<arma::vec, arma::vec>
{
    static void resize(arma::vec& v1, const arma::vec& v2)
    {
        v1.set_size(v2.n_elem);
    }
};

// ===================== arma::rowvec =====================
template <>
struct is_resizeable<arma::rowvec> : boost::true_type {};

template <>
struct same_size_impl<arma::rowvec, arma::rowvec>
{
    static bool same_size(const arma::rowvec& x, const arma::rowvec& y)
    {
        return x.n_elem == y.n_elem;
    }
};

template <>
struct resize_impl<arma::rowvec, arma::rowvec>
{
    static void resize(arma::rowvec& v1, const arma::rowvec& v2)
    {
        v1.set_size(1, v2.n_elem);
    }
};

// ===================== arma::cx_vec =====================
template <>
struct is_resizeable<arma::cx_vec> : boost::true_type {};

template <>
struct same_size_impl<arma::cx_vec, arma::cx_vec>
{
    static bool same_size(const arma::cx_vec& x, const arma::cx_vec& y)
    {
        return x.n_elem == y.n_elem;
    }
};

template <>
struct resize_impl<arma::cx_vec, arma::cx_vec>
{
    static void resize(arma::cx_vec& v1, const arma::cx_vec& v2)
    {
        v1.set_size(v2.n_elem);
    }
};

} } }

template <>
struct boost::numeric::odeint::vector_space_norm_inf<arma::vec> {
    static double norm_inf(const arma::vec &x) {
        return arma::abs(x).max();
    }
};

class NonResonantDriving{
private:
    PolaritonMode* polariton;
    double coupling_constant;
    double time_factor;
    double power;
    double alpha;
    double value;

public:
    NonResonantDriving(PolaritonMode* polariton_mode, 
                       double coupling, double tau, double P, 
                       double alpha_, double n0)
        : polariton(polariton_mode)
        , coupling_constant(coupling)
        , time_factor(tau)
        , power(P)
        , alpha(alpha_)
        , value(n0)
    {}

    double get_value() const { return value; }
    void set_value(double n) { value = n; }
    void set_power(double P) { power = P; }
    
    double derivative(double t) const;
};

class PolaritonMode{
private:
    std::vector<PolaritonMode*> neighboring_polaritons;
    std::vector<PhononMode*> neighboring_phonons;
    std::vector<arma::cx_double> constant_couplings;
    std::vector<arma::cx_double> phonon_couplings;
    std::vector<double> detunings_from_resonance;
    std::vector<double> signs;
    size_t n_neighbors;
    
    arma::cx_double r_driving_amp;
    double r_driving_detuning;
    
    // Store reservoir directly with ownership
    std::unique_ptr<NonResonantDriving> reservoir_owned;
    double reservoir_coupling;
    
    double dissipative_gamma;
    double self_interaction;
    arma::cx_double value;

public:
    PolaritonMode(double gamma, double U)
        : dissipative_gamma(gamma)
        , self_interaction(U)
        , n_neighbors(0)
        , r_driving_amp(0.0)
        , r_driving_detuning(0.0)
        , reservoir_coupling(0.0)
        , value(0.0)
    {}

    void connect(PolaritonMode* polariton, PhononMode* phonon, 
                 double J, double g, double delta, bool above = true){
        neighboring_polaritons.push_back(polariton);
        neighboring_phonons.push_back(phonon);
        constant_couplings.push_back(J);
        phonon_couplings.push_back(g);
        detunings_from_resonance.push_back(delta);
        signs.push_back(above ? 1.0 : -1.0);
        n_neighbors += 1;
    }

    // Simple add_reservoir that works before Cavity construction
    void add_reservoir(double coupling, double tau, double pump_power, double alpha, double initial_n = 0.0){
        reservoir_owned = std::make_unique<NonResonantDriving>(
            this, coupling, tau, pump_power, alpha, initial_n
        );
        reservoir_coupling = coupling;
    }

    NonResonantDriving* get_reservoir() const { 
        return reservoir_owned.get(); 
    }   
    
    arma::cx_double get_value() const { return value; }
    void set_value(arma::cx_double val) { value = val; }
    
    void set_driving(arma::cx_double amp, double detuning){
        r_driving_amp = amp;
        r_driving_detuning = detuning;
    }

    void check(){
        std::vector<size_t> dimensions = {
            neighboring_polaritons.size(),
            neighboring_phonons.size(),
            constant_couplings.size(),
            phonon_couplings.size(),
            detunings_from_resonance.size(),
            signs.size()
        };
        
        if(!std::all_of(dimensions.begin(), dimensions.end(), 
                        [&](size_t x){return x == dimensions.front();})){
            throw std::domain_error("All system arrays must have the same dimensions");
        }
        n_neighbors = dimensions[0];
    }
    
    arma::cx_double derivative(double t) const;
};

class PhononMode{
private: 
    std::vector<std::array<PolaritonMode*, 2>> pairs;
    std::vector<double> detunings;
    std::vector<double> couplings;
    size_t n_pairings;
    double frequency;
    double dissipation;
    double position;
    double velocity;

public:
    PhononMode(double Omega, double Gamma)
        : frequency(Omega)
        , dissipation(Gamma)
        , n_pairings(0)
        , position(0.0)
        , velocity(0.0)
    {}

    void add_pairing(std::array<PolaritonMode*,2> polariton_pair, 
                     double delta, double coupling){
        pairs.push_back(polariton_pair);
        detunings.push_back(delta);
        couplings.push_back(coupling);
        n_pairings += 1;
    }

    void check(){
        std::vector<size_t> dimensions = {
            pairs.size(),
            detunings.size(),
            couplings.size()
        };
        
        if(!std::all_of(dimensions.begin(), dimensions.end(), 
                        [&](size_t x){return x == dimensions.front();})){
            throw std::domain_error("All system arrays must have the same dimensions");
        }
        n_pairings = dimensions[0];
    }

    double get_position() const { return position; }
    double get_velocity() const { return velocity; }
    double get_freq() const { return frequency; }
    
    void set_position(double x) { position = x; }
    void set_velocity(double v) { velocity = v; }
    
    double second_derivative(double t) const;
};

class Cavity{
private:
    std::vector<PolaritonMode*> polaritons;
    std::vector<PhononMode*> phonons;
    arma::vec current_state;
    double current_time;
    stationary_stepper rk4_stepper;
    adaptive_stepper dopri5_stepper;
    double abs_stepper_tol = 1e-6;
    double rel_stepper_tol = 1e-6;
    double used_time_step = 1e-3;
    size_t system_dimension;
    bool evaluating_rhs;

    friend class CavityConfig;
public:
    // Delete copy constructor and copy assignment
    Cavity(const Cavity&) = delete;
    Cavity& operator=(const Cavity&) = delete;
    
    // Allow move constructor and move assignment
    Cavity(Cavity&&) = default;
    Cavity& operator=(Cavity&&) = default;

    // Default constructor
    Cavity() : current_time(0.0), system_dimension(0), evaluating_rhs(false) {
        dopri5_stepper = boost::numeric::odeint::make_controlled<error_stepper>(abs_stepper_tol, rel_stepper_tol);
    }
    
    // Main constructor - automatically handles reservoirs
    Cavity(std::vector<PolaritonMode*> polariton_modes, 
           std::vector<PhononMode*> phonon_modes, 
           double t0)
        : polaritons(std::move(polariton_modes))
        , phonons(std::move(phonon_modes))
        , current_time(t0)
        , system_dimension(0)
        , evaluating_rhs(false)
    {   
        dopri5_stepper = boost::numeric::odeint::make_controlled<error_stepper>(abs_stepper_tol, rel_stepper_tol);
        for(auto* p : polaritons) p->check();
        for(auto* ph : phonons) ph->check();
        
        size_t N = 2 * polaritons.size() + 2 * phonons.size();
        
        // Count reservoirs that were added before construction
        for(auto* p : polaritons){
            if(p->get_reservoir() != nullptr){
                N += 1;
            } 
        }
        
        system_dimension = N;
        current_state.set_size(system_dimension);
        pack_state();
    }
    
    ~Cavity() = default;
    PolaritonMode* get_polariton(size_t index){
        if(index >= polaritons.size()){
            throw std::range_error("Polariton index out of bounds");
        }
        return polaritons[index];
    }
    PhononMode* get_phonon(size_t index){
        if(index >= phonons.size()){
            throw std::range_error("Phonon index out of bounds");
        }
        return phonons[index];
    }
    // void load_from_file(const std::string& filename, double t0 = 0.0){
    //     std::ifstream file(filename);
    //     if(!file.is_open()){
    //         throw std::runtime_error("Cannot open file: " + filename);
    //     }

    //     // Clean previous system (Cavity owns these)
    //     for(auto* p : polaritons) delete p;
    //     for(auto* ph : phonons) delete ph;
    //     polaritons.clear();
    //     phonons.clear();

    //     // RNG
    //     std::mt19937_64 rng(std::random_device{}());
    //     auto uniform_sym = [&](double A){
    //         std::uniform_real_distribution<double> d(-A, A);
    //         return d(rng);
    //     };
    //     auto uniform_pos = [&](double A){
    //         std::uniform_real_distribution<double> d(0.0, A);
    //         return d(rng);
    //     };

    //     // Random flags
    //     bool random_polariton = false;
    //     bool random_phonon    = false;
    //     bool random_reservoir = false;
    //     double pol_amp = 0.0;
    //     double ph_amp  = 0.0;
    //     double res_amp = 0.0;

    //     std::string line;
    //     size_t line_number = 0;

    //     while(std::getline(file, line)){
    //         ++line_number;

    //         // Strip comments
    //         if(auto pos = line.find('#'); pos != std::string::npos)
    //             line = line.substr(0, pos);

    //         // Trim
    //         line.erase(0, line.find_first_not_of(" \t\r\n"));
    //         if(line.empty()) continue;

    //         std::istringstream iss(line);
    //         std::string keyword;
    //         iss >> keyword;

    //         try {

    //             if(keyword == "POLARITON"){
    //                 size_t id; double gamma, U;
    //                 iss >> id >> gamma >> U;

    //                 if(id >= polaritons.size())
    //                     polaritons.resize(id + 1, nullptr);

    //                 polaritons[id] = new PolaritonMode(gamma, U);
    //             }

    //             else if(keyword == "PHONON"){
    //                 size_t id; double omega, Gamma;
    //                 iss >> id >> omega >> Gamma;

    //                 if(id >= phonons.size())
    //                     phonons.resize(id + 1, nullptr);

    //                 phonons[id] = new PhononMode(omega, Gamma);
    //             }

    //             else if(keyword == "CONNECT"){
    //                 size_t p, q, ph; double J, g, d; int above;
    //                 iss >> p >> q >> ph >> J >> g >> d >> above;

    //                 polaritons[p]->connect(
    //                     polaritons[q], phonons[ph], J, g, d, above != 0
    //                 );
    //             }

    //             else if(keyword == "PAIRING"){
    //                 size_t ph, up, lo; double g, d;
    //                 iss >> ph >> up >> lo >> g >> d;

    //                 phonons[ph]->add_pairing({polaritons[up], polaritons[lo]}, d, g);
    //             }

    //             else if(keyword == "DRIVING"){
    //                 size_t p; double re, im, d;
    //                 iss >> p >> re >> im >> d;
    //                 polaritons[p]->set_driving({re, im}, d);
    //             }

    //             else if(keyword == "RESERVOIR"){
    //                 size_t p; double c, tau, P, a, n0 = 0.0;
    //                 iss >> p >> c >> tau >> P >> a;
    //                 iss >> n0;

    //                 polaritons[p]->add_reservoir(c, tau, P, a, n0);
    //             }

    //             else if(keyword == "INITIAL"){
    //                 size_t p; double re, im;
    //                 iss >> p >> re >> im;
    //                 polaritons[p]->set_value({re, im});
    //             }

    //             else if(keyword == "PHONON_INITIAL"){
    //                 size_t ph; double x, v;
    //                 iss >> ph >> x >> v;
    //                 phonons[ph]->set_position(x);
    //                 phonons[ph]->set_velocity(v);
    //             }

    //             // -------- RANDOM INITIAL CONDITIONS --------

    //             else if(keyword == "RANDOM_POLARITON"){
    //                 iss >> pol_amp;
    //                 random_polariton = true;
    //             }

    //             else if(keyword == "RANDOM_PHONON"){
    //                 iss >> ph_amp;
    //                 random_phonon = true;
    //             }

    //             else if(keyword == "RANDOM_RESERVOIR"){
    //                 iss >> res_amp;
    //                 random_reservoir = true;
    //             }

    //             else {
    //                 throw std::runtime_error("Unknown keyword: " + keyword);
    //             }

    //         }
    //         catch(const std::exception& e){
    //             throw std::runtime_error(
    //                 "Error on line " + std::to_string(line_number) + ": " + e.what()
    //             );
    //         }
    //     }

    //     // Remove sparse nullptrs
    //     polaritons.erase(std::remove(polaritons.begin(), polaritons.end(), nullptr),
    //                     polaritons.end());
    //     phonons.erase(std::remove(phonons.begin(), phonons.end(), nullptr),
    //                 phonons.end());

    //     // Apply random initial conditions
    //     if(random_polariton){
    //         for(auto* p : polaritons){
    //             p->set_value({
    //                 uniform_sym(pol_amp),
    //                 uniform_sym(pol_amp)
    //             });
    //         }
    //     }

    //     if(random_phonon){
    //         for(auto* ph : phonons){
    //             ph->set_position(uniform_sym(ph_amp));
    //             ph->set_velocity(uniform_sym(ph_amp));
    //         }
    //     }

    //     if(random_reservoir){
    //         for(auto* p : polaritons){
    //             if(auto* r = p->get_reservoir()){
    //                 double n = r->get_value() + uniform_pos(res_amp);
    //                 r->set_value(std::max(0.0, n));
    //             }
    //         }
    //     }

    //     // Final initialization
    //     current_time = t0;
    //     dopri5_stepper = boost::numeric::odeint::make_controlled<error_stepper>(
    //         abs_stepper_tol, rel_stepper_tol
    //     );

    //     for(auto* p : polaritons) p->check();
    //     for(auto* ph : phonons) ph->check();

    //     size_t N = 2 * polaritons.size() + 2 * phonons.size();
    //     for(auto* p : polaritons)
    //         if(p->get_reservoir()) ++N;

    //     system_dimension = N;
    //     current_state.set_size(system_dimension);
    //     pack_state();

    //     std::cout << "Successfully loaded system:\n"
    //             << "  " << polaritons.size() << " polaritons\n"
    //             << "  " << phonons.size() << " phonons\n"
    //             << "  dimension = " << system_dimension << std::endl;
    // }


    void pack_state(){
        size_t idx = 0;
        
        for(const auto* p : polaritons){
            arma::cx_double v = p->get_value();
            current_state(idx) = v.real();
            current_state(idx + 1) = v.imag();
            idx += 2;
        }
        
        for(const auto* ph : phonons){
            current_state(idx) = ph->get_position();
            current_state(idx + 1) = ph->get_velocity();
            idx += 2;
        }
        
        // Pack reservoirs from polaritons
        for(const auto* p : polaritons){
            if(p->get_reservoir() != nullptr){
                current_state(idx) = p->get_reservoir()->get_value();
                idx += 1;
            }
        }
    }

    void unpack_state(const arma::vec& x){
        size_t idx = 0;

        for (auto* p : polaritons) {
            arma::cx_double v(x(idx), x(idx + 1));
            p->set_value(v);
            idx += 2;
        }

        for (auto* ph : phonons) {
            ph->set_position(x(idx));
            ph->set_velocity(x(idx + 1));
            idx += 2;
        }

        // Unpack reservoirs
        for (auto* p : polaritons) {
            if(p->get_reservoir() != nullptr){
                p->get_reservoir()->set_value(x(idx));
                idx += 1;
            }
        }
    }
    double get_time_step(){
        return used_time_step;
    }
    void do_step(double dt){
        rk4_stepper.do_step(std::ref(*this), current_state, current_time, dt);
        unpack_state(current_state);
        used_time_step = dt;
    }
    
    void adaptive_step(){
    dopri5_stepper.try_step(std::ref(*this), current_state, current_time, used_time_step);
    unpack_state(current_state);
    }


    void operator()(const arma::vec& x, arma::vec& dxdt, const double t){
        dxdt.set_size(system_dimension);
        
        evaluating_rhs = true;
        unpack_state(x);

        size_t idx = 0;
        arma::cx_double d;
        
        for (auto* p : polaritons) {
            d = p->derivative(t);
            dxdt(idx)     = d.real();
            dxdt(idx + 1) = d.imag();
            idx += 2;
        }

        for (auto* ph : phonons) {
            dxdt(idx)     = ph->get_velocity();
            dxdt(idx + 1) = ph->second_derivative(t);
            idx += 2;
        }

        // Reservoir derivatives
        for (auto* p : polaritons) {
            if(p->get_reservoir() != nullptr){
                dxdt(idx) = p->get_reservoir()->derivative(t);
                idx += 1;
            }
        }
        
        evaluating_rhs = false;
    }

    const arma::vec& get_state() const { return current_state; }
    double get_time() const { return current_time; }
};

// Out-of-line definitions

inline arma::cx_double PolaritonMode::derivative(double t) const {
    arma::cx_double res = reservoir_owned ? reservoir_owned->get_value() : 0.0;
    arma::cx_double drv = value * (-I * dissipative_gamma + 
                                   self_interaction * value * conj(value) + 
                                   I * reservoir_coupling * res) + 
                         r_driving_amp * std::polar(1.0, r_driving_detuning * t);
    
    for(size_t i = 0; i < n_neighbors; ++i){
        const double phase = signs[i] * (neighboring_phonons[i]->get_freq() + 
                                         detunings_from_resonance[i])*t;
        const arma::cx_double phonon_term = 
            phonon_couplings[i] *  neighboring_phonons[i]->get_position();
        
        drv += (constant_couplings[i] + phonon_term) * std::polar(1.0, phase) *
               neighboring_polaritons[i]->get_value();
    }
    
    return -I * drv;
}

inline double PhononMode::second_derivative(double t) const {
    double drv = -frequency * frequency * position - dissipation * velocity;
    arma::cx_double backaction(0.0, 0.0);
    
    for(size_t i = 0; i < n_pairings; ++i){
        const arma::cx_double phase_factor = 
            std::polar(1.0, -(frequency + detunings[i]) * t);
        backaction += couplings[i] * 
                      pairs[i][0]->get_value() * 
                      conj(pairs[i][1]->get_value()) * 
                      phase_factor;
    }
    
    drv += -2.0 * frequency * dissipation * backaction.real();
    return drv;
}

inline double NonResonantDriving::derivative(double t) const {
    const double intensity = (polariton->get_value() * 
                             conj(polariton->get_value())).real();
    return time_factor * (power - value * (1.0 + alpha * alpha * intensity));
}