#include "polariton.h"


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
    
    // Set initial values
    site_1.set_value(arma::cx_double{uniform(rng), uniform(rng)});
    site_1.add_reservoir(1.0, 600, driving_value, sqrt(3.25), 0.1 + 0.5 * uniform(rng));
    site_2.set_value(10.0*arma::cx_double{uniform(rng), uniform(rng)});
    site_2.add_reservoir(1.0, 600, 0.5*driving_value, sqrt(3.25), 0.1 + 0.5 * uniform(rng));
    // site_1.set_driving(1.0, 0.6);
    // Add reservoir directly to polariton
    
    // site_2.add_reservoir(1.0, 1.0, 6.0, 2.1, 3.0);
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
    site_1.check();
    site_2.check();
    // Create cavity - it automatically handles everything!
    std::vector<PolaritonMode*> sites = {&site_1, &site_2};
    std::vector<PhononMode*> phonons = {&phonon};
    Cavity model(sites, phonons, 0.0);
    model.pack_state();
    std::cout << "System dimension: " << model.get_state().n_rows << std::endl;
    std::cout << "Initial state:" << std::endl;
    for(size_t i = 0; i < model.get_state().n_rows; ++i){
        std::cout << "  state[" << i << "] = " << model.get_state()(i) << std::endl;
    }
    std::cout << "site_1 value: " << site_1.get_value() << std::endl;
    std::cout << "site_2 value: " << site_2.get_value() << std::endl;
    std::cout << "reservoir: " << (site_1.get_reservoir() ? site_1.get_reservoir()->get_value() : -999) << std::endl;
    size_t dimension = model.get_state().n_rows;
    
      /*
    ============ Integrator setup ============
    */
    size_t transient = 1e7;
    size_t stationary = 5e5;
    
    /*
    ============ Evolution and storage ============
    */
    // Transient phase
    // 
    // arma::vec aux;
    for(size_t i = 0; i < transient + stationary; i++){  // Changed variable name
        model.adaptive_step();
    }
    // Stationary phase - compute averages
    std::vector<double> avg{0.0, 0.0, 0.0, 0.0, 0.0};
    arma::vec aux;
    double dt = model.get_time_step();
    for(size_t i = 0; i < stationary; i++){  // Changed variable name
        aux = model.get_state();
        avg[0] += aux(0)*aux(0) + aux(1)*aux(1);  // |site_1|^2
        avg[1] += aux(2)*aux(2) + aux(3)*aux(3);  // |site_2|^2
        avg[2] += aux(4)*aux(4);                   // phonon position^2
        avg[3] += aux(6);
        avg[4] += aux(7);
        model.adaptive_step();
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