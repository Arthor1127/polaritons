#include "polariton.h"
#include "cavity_config.h"

int main(int argc, char* argv[]){
    if(argc != 4){
        std::cout << "Usage: ./program <steps> <index> <output_file>" << std::endl;
        return 1;  // Exit if wrong number of arguments
    }
    size_t steps = std::stoi(argv[1]);
    size_t job_index = std::stoi(argv[2]);  // Renamed to avoid collision
    auto file_name = argv[3];
    
    Cavity model;
    CavityConfig config;
    try{
        config.load_from_ini(model, "/home/ruiz/Documents/polaritons/scripts/cpp/two_site_non_resonant.build");
    }
    catch(const std::exception& e){
        std::cerr << "Error loading configuration: " << e.what() << std::endl;
        return 1;
    }
    
    double driving_start = 0.0;
    double driving_stop = 15.0;
    double power = arma::linspace(driving_start, driving_stop, steps)[job_index];
    size_t site1_id = config.get_polariton_id("site_1");
    size_t site2_id = config.get_polariton_id("site_2");

    // model.get_polariton(site1_id)->get_reservoir()->set_power(power);
    model.get_polariton(site1_id)->set_driving(0.5*power, 0.0);
    model.get_polariton(site2_id)->get_reservoir()->set_power(0.5 * power);
    // std::mt19937 gen(std::random_device{}());
    // std::uniform_real_distribution<double> dist(0.0, 1.0);

    // auto rand_phase = [&]() {
    //     return std::polar(1.0, 2.0 * M_PI * dist(gen));
    // };
    // double alpha_2 = 3.25;
    // double P0 = 1.0 + alpha_2;

    // auto* p1 = model.get_polariton(config.get_polariton_id("site_1"));
    // auto* p2 = model.get_polariton(config.get_polariton_id("site_2"));
    // auto* ph = model.get_phonon(0);

    // auto* r1 = p1->get_reservoir();

    // // ===== Regime selection =====
    // if (power <= 1.0) {
    //     p1->set_value(1e-3 * dist(gen) * rand_phase());
    //     p2->set_value(1e-3 * dist(gen) * rand_phase());

    //     ph->set_position(1e-3 * dist(gen));
    //     ph->set_velocity(dist(gen) * ph->get_position());

    //     r1->set_value(power * (1.0 - 0.2 * dist(gen)));
    // }
    // else if (power <= P0) {
    //     p1->set_value((1.0 + 0.1 * dist(gen)) *
    //                 rand_phase() * (power - 1.0) / alpha_2);

    //     p2->set_value(1e-3 * dist(gen) * rand_phase());

    //     ph->set_position(1e-3 * dist(gen));
    //     ph->set_velocity(dist(gen) * ph->get_position());

    //     r1->set_value(1.0 - 0.2 * dist(gen));
    // }
    // else {
    //     p1->set_value((1.0 + 0.1 * dist(gen)) * rand_phase());

    //     p2->set_value((1.0 + 0.1 * dist(gen)) *
    //                 rand_phase() * (power / P0 - 1.0));

    //     ph->set_position((0.1 * dist(gen) + 1.0) *
    //                     std::sqrt(power / P0 - 1.0));

    //     ph->set_velocity((2.0 * dist(gen) - 1.0) *
    //                     ph->get_position());

    //     r1->set_value(1.0 - 0.1 * dist(gen));
    // }


    model.pack_state();

    std::cout << "System dimension: " << model.get_state().n_rows << std::endl;
    std::cout << "Initial state:" << std::endl;
    for(size_t i = 0; i < model.get_state().n_rows; ++i){
        std::cout << "  state[" << i << "] = " << model.get_state()(i) << std::endl;
    }
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
    for(size_t i = 0; i < transient; i++){  // Changed variable name
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
        // avg[4] += aux(7);
        model.adaptive_step();
    }
    
    // Write output
    std::ofstream output_file(file_name);
    output_file << power << "\t";
    for(auto& e : avg){
        output_file << e/static_cast<double>(stationary) << "\t"; 
    }
    output_file << std::endl;
    output_file.close();
    return 0;
}   