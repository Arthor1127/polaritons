/*
Header for programs used in the analysis of compilation

The flow of the program would be the following. In the system there are three main classes.

class PolaritonMode
class PhononMode
class NonResonantDriving

which are to be combined in a single class (Cavity)

The idea is that given a set of Polariton, Phonon and driving objects, I can construct the Cavity class and this last one constructs the system of differential equations to be integrated and the jacobian.

This approach would later prove useful for the implementation of live parameter modification and analysis and also for the implementation of different models like multiple polaritons coupled via a single phonon. 

Also the idea is that the Cavity class can be built from a.txt file to be passed to the main program and then to the class constructor. 

Briefly, I'm implementing a driven-dissipative tight-binding time resolved solver with the possibility of introducing time dependent (and system responsive) couplings.
*/
#pragma once
#include<functional>
#include<iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>
#include<armadillo>
#include<boost/numeric/odeint.hpp>
#include <fftw3.h>

constexpr arma::cx_double I(0.0, 1.0);
using stationary_stepper = boost::numeric::odeint::runge_kutta4<arma::vec>;
using adaptive_stepper = boost::numeric::odeint::runge_kutta_dopri5<arma::vec>;

class PolaritonMode;
class PhononMode;
class NonResonantDriving;

// namespace redefinitions for the compatibility of armadillo with odeint.
namespace boost { namespace numeric { namespace odeint {

// ===================== arma::arma::vec =====================
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

// Probably is better to define the non rotating frame equations in the interface and the rotating frame ones in the implementation
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
    NonResonantDriving* reservoir = nullptr;
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

    // Add this method - it was missing!
    void set_reservoir(NonResonantDriving* n, double coupling) {
        reservoir = n;
        reservoir_coupling = coupling;
    }

    NonResonantDriving* get_reservoir() const { return reservoir; }
    
    // Add const qualifier here!
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

    // Add const qualifiers here!
    double get_position() const { return position; }
    double get_velocity() const { return velocity; }
    double get_freq() const { return frequency; }
    
    void set_position(double x) { position = x; }
    void set_velocity(double v) { velocity = v; }
    
    double second_derivative(double t) const;
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

class Cavity{
private:
    std::vector<PolaritonMode*> polaritons;
    std::vector<std::unique_ptr<NonResonantDriving>> polariton_reservoirs;
    std::vector<PhononMode*> phonons;
    arma::vec current_state;
    double current_time;
    stationary_stepper stepper;
    size_t system_dimension;
    bool evaluating_rhs;  // Fixed: declaration, not assignment

public:
    // Delete copy constructor and copy assignment
    Cavity(const Cavity&) = delete;
    Cavity& operator=(const Cavity&) = delete;
    
    // Allow move constructor and move assignment
    Cavity(Cavity&&) = default;
    Cavity& operator=(Cavity&&) = default;

    // Default constructor
    Cavity() : current_time(0.0), system_dimension(0), evaluating_rhs(false) {}
    
    // Main constructor
    Cavity(std::vector<PolaritonMode*> polariton_modes, 
           std::vector<PhononMode*> phonon_modes, 
           double t0)
        : polaritons(std::move(polariton_modes))
        , phonons(std::move(phonon_modes))
        , current_time(t0)
        , system_dimension(0)
        , evaluating_rhs(false)
    {
        for(auto* p : polaritons) p->check();
        for(auto* ph : phonons) ph->check();
        
        size_t N = 2 * polaritons.size() + 2 * phonons.size();
        
        for(auto* p : polaritons){
            if(p->get_reservoir() != nullptr){
                N += 1;
            } 
        }
        
        system_dimension = N;
        current_state.set_size(system_dimension);
        pack_state();
    }
    
    // Destructor - unique_ptr auto-deletes, so just default
    ~Cavity() = default;

    // Factory method for creating reservoirs with proper ownership
    NonResonantDriving* create_reservoir(PolaritonMode* pol, 
                                        double coupling, 
                                        double tau, 
                                        double pump_power, 
                                        double alpha){
        auto reservoir = std::make_unique<NonResonantDriving>(
            pol, coupling, tau, pump_power, alpha, 0.0
        );
        NonResonantDriving* ptr = reservoir.get();
        polariton_reservoirs.push_back(std::move(reservoir));
        pol->set_reservoir(ptr, coupling);
        return ptr;
    }

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
        
        for(const auto& r : polariton_reservoirs){
            current_state(idx) = r->get_value();
            idx += 1;
        }
    }

    void load_from_document(){
        // To be implemented
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

        for (auto& r : polariton_reservoirs) {
            r->set_value(x(idx));
            idx += 1;
        }
    }

    void do_step(double& t, double dt){
        stepper.do_step(std::ref(*this), current_state, t, dt);
        unpack_state(current_state);
        t += dt;
        current_time = t;
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

        for (const auto& r : polariton_reservoirs) {
            dxdt(idx) = r->derivative(t);
            idx += 1;
        }
        
        evaluating_rhs = false;
    }

    const arma::vec& get_state() const { return current_state; }
    double get_time() const { return current_time; }
};