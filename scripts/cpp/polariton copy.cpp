#include "polariton.h"
#include <fstream>
#include <random>

std::mt19937 rng{ std::random_device{}() };
std::uniform_real_distribution<double> uniform(0.0, 1.0);

int main(){
    /*
    ============ Cavity Setup ============
    */
    PolaritonMode site_1(1.0, 0.0);
    PolaritonMode site_2(1.0, 0.0);
    
    // Set initial values
    site_1.set_value(arma::cx_double{uniform(rng), uniform(rng)});
    site_2.set_value(arma::cx_double{uniform(rng), uniform(rng)});
    site_1.set_driving(1.0, 0.6);
    // Add reservoir directly to polariton
    site_1.add_reservoir(1.0, 1.0, 12.0, sqrt(3.25), uniform(rng));
    site_2.add_reservoir(1.0, 1.0, 6.0, 2.1, 3.0);
    // Setup phonon
    PhononMode phonon(20.0, 0.05);
    // phonon.set_position(10.0 * uniform(rng));
    // phonon.set_velocity(200.0 * uniform(rng));
    phonon.set_position(10.0 * uniform(rng));
    phonon.set_velocity(200.0 * uniform(rng));
    // Connect sites
    site_1.connect(&site_2, &phonon, 0.0, 1.0, 0.0, true);
    site_2.connect(&site_1, &phonon, 0.0, 1.0, 0.0, false);
    phonon.add_pairing({&site_1, &site_2}, 0.0, 1.0);
    
    // Create cavity - it automatically handles everything!
    std::vector<PolaritonMode*> sites = {&site_1, &site_2};
    std::vector<PhononMode*> phonons = {&phonon};
    Cavity model(sites, phonons, 0.0);
    
    size_t dimension = model.get_state().n_rows;
    
    /*
    ============ Integration ============
    */
    size_t n_steps = 1e5;
    double delta_t = 0.005;
    
    std::ofstream output_file("/home/ruiz/Documents/polaritons/data/raw/two_site_non_resonant.dat");
    
    for(size_t index = 0; index < n_steps; index++){
        output_file << model.get_time() << "\t";
        for(size_t i = 0; i < dimension; i++){
            output_file << model.get_state()(i);
            if(i != dimension - 1) output_file << "\t";
        }
        output_file << "\n";
        model.do_step(delta_t);
    }
    
    output_file.close();
    std::cout << "Simulation complete!" << std::endl;
    
    return 0;
}