#include "polariton.h"
#include <fstream>
#include <random>

std::mt19937 rng{std::random_device{}()};
std::uniform_real_distribution<double> uniform(0.0, 1.0);

int main(int argc, char* argv[]){
    if(argc != 4){
        std::cout << "Usage: ./program <steps> <index> <output_file>" << std::endl;
        return 1;  // Exit if wrong number of arguments
    }
    
    size_t steps = std::stoi(argv[1]);
    size_t job_index = std::stoi(argv[2]);  // Renamed to avoid collision
    auto file_name = argv[3];
    
    double driving_start = 0.0;
    double driving_stop = 14.0;
    double driving_value = arma::linspace(driving_start, driving_stop, steps)[job_index];
    
    /*
    ============ Cavity Setup ============
    */
    PolaritonMode site_1(1.0, 0.0);
    PolaritonMode site_2(1.0, 0.0);
    
    site_1.set_value(arma::cx_double{uniform(rng), uniform(rng)});
    site_2.set_value(arma::cx_double{uniform(rng), uniform(rng)});  // Fixed: was site_1
    
    site_1.set_driving(0.0, 0.0);
    site_2.set_driving(0.0, 0.0);
    PhononMode phonon(20.0, 0.05);
    phonon.set_position(50.0 * uniform(rng));
    phonon.set_velocity(200.0 * uniform(rng));
    
    site_1.connect(&site_2, &phonon, 10.0, 1.0, 0.0, true);
    site_2.connect(&site_1, &phonon, 10.0, 1.0, 0.0, false);
    
    // Add phonon pairing - IMPORTANT!
    phonon.add_pairing({&site_1, &site_2}, 0.0, 1.0);
    
    std::vector<PolaritonMode*> sites = {&site_1, &site_2};
    std::vector<PhononMode*> phonons = {&phonon};
    
    Cavity model(sites, phonons, 0.0);
    size_t dimension = model.get_state().n_rows;
    
    /*
    ============ Integrator setup ============
    */
    size_t transient = 5e5;
    size_t stationary = 1e3;
    double delta_t = 0.005;
    
    /*
    ============ Evolution and storage ============
    */
    // Transient phase
    for(size_t i = 0; i < transient; i++){  // Changed variable name
        model.do_step(delta_t);
    }
    
    // Stationary phase - compute averages
    std::vector<double> avg{0.0, 0.0, 0.0};
    arma::vec aux;
    
    for(size_t i = 0; i < stationary; i++){  // Changed variable name
        aux = model.get_state();
        avg[0] += aux(0)*aux(0) + aux(1)*aux(1);  // |site_1|^2
        avg[1] += aux(2)*aux(2) + aux(3)*aux(3);  // |site_2|^2
        avg[2] += aux(4)*aux(4);                   // phonon position^2
        model.do_step(delta_t);
    }
    
    // Write output
    std::ofstream output_file(file_name);
    output_file << driving_value << "\t";
    for(auto& e : avg){
        output_file << e/static_cast<double>(stationary) << "\t"; 
    }
    output_file << std::endl;
    output_file.close();
    
    return 0;
}