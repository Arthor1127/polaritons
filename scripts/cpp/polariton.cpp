#include "polariton.h"
#include<random>
std::mt19937 rng{ std::random_device{}() };
std::uniform_real_distribution<double> uniform(0.0, 1.0);
int main(){
    // Implementation with two sites with the first resonantly driven

    /*
    ============ Cavity Setup ============
    */
    PolaritonMode site_1(1.0, 0.0);
    PolaritonMode site_2(1.0, 0.0);
    site_1.set_value(arma::cx_double{uniform(rng), uniform(rng)});
    site_1.set_value(arma::cx_double{uniform(rng), uniform(rng)});
    site_1.set_driving(1.5, 0.0);
    PhononMode phonon(20.0, 0.05);
    phonon.set_position(10.0 * uniform(rng));
    phonon.set_velocity(200.0 * uniform(rng));
    site_1.connect(&site_2, &phonon, 5.0, 1.0, 0.0, true);
    site_2.connect(&site_1, &phonon, 5.0, 1.0, 0.0, false);
    std::vector<PolaritonMode*> sites = {&site_1, &site_2};
    std::vector<PhononMode*> phonons = {&phonon};
    stationary_stepper integrator;
    Cavity model(sites, phonons, 0.0);
    size_t dimension = model.get_state().n_rows;
    /*
    ============ Integrator setup ============
    */
    size_t n_steps = 1e4;
    double delta_t = 0.005;
    double t = 0.0;
    /*
    ============ Evolution and storage ============
    */
    std::ofstream output_file("/home/ruiz/Documents/polaritons/data/raw/two_site_resonant.dat");
    for(size_t index = 0; index < n_steps; index++){
        for(size_t element_index = 0; element_index < dimension; element_index++){
            if(element_index != dimension-1){
                output_file << model.get_state()(element_index) << "\t";
            }
            else{
                output_file << model.get_state()(element_index) << "\n";
            }
            model.do_step(delta_t);
        }
    }
    output_file.close();
}