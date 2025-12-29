/*
INI-style configuration parser for Cavity class
Supports named entities, expressions, and random initial conditions
*/

#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <fstream>
#include <random>
#include <functional>
#include <cmath>
#include <iostream>

class ConfigParser {
private:
    std::mt19937 rng;
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> sections;
    
    // Parse expressions like "uniform(0.0, 1.0)"
    double parse_expression(const std::string& expr) {
        std::string trimmed = trim(expr);
        
        // Check if it's a number
        try {
            return std::stod(trimmed);
        } catch(...) {}
        
        // Check for uniform(a, b)
        if(trimmed.substr(0, 7) == "uniform" || trimmed.substr(0, 7) == "Uniform"){
            size_t start = trimmed.find('(');
            size_t comma = trimmed.find(',');
            size_t end = trimmed.find(')');
            
            if(start != std::string::npos && comma != std::string::npos && end != std::string::npos){
                double a = std::stod(trimmed.substr(start + 1, comma - start - 1));
                double b = std::stod(trimmed.substr(comma + 1, end - comma - 1));
                std::uniform_real_distribution<double> dist(a, b);
                return dist(rng);
            }
        }
        
        // Check for normal(mean, std)
        if(trimmed.substr(0, 6) == "normal" || trimmed.substr(0, 6) == "Normal"){
            size_t start = trimmed.find('(');
            size_t comma = trimmed.find(',');
            size_t end = trimmed.find(')');
            
            if(start != std::string::npos && comma != std::string::npos && end != std::string::npos){
                double mean = std::stod(trimmed.substr(start + 1, comma - start - 1));
                double stddev = std::stod(trimmed.substr(comma + 1, end - comma - 1));
                std::normal_distribution<double> dist(mean, stddev);
                return dist(rng);
            }
        }
        
        throw std::runtime_error("Cannot parse expression: " + expr);
    }
    
    std::string trim(const std::string& s) const {
        auto start = s.find_first_not_of(" \t\r\n");
        if(start == std::string::npos) return "";
        auto end = s.find_last_not_of(" \t\r\n");
        return s.substr(start, end - start + 1);
    }
    
    std::vector<std::string> split(const std::string& s, char delim) const {
        std::vector<std::string> result;
        std::stringstream ss(s);
        std::string item;
        while(std::getline(ss, item, delim)){
            result.push_back(trim(item));
        }
        return result;
    }

public:
    ConfigParser() {
        rng.seed(std::random_device{}());
    }
    
    void set_seed(unsigned int seed) {
        rng.seed(seed);
    }
    
    void load(const std::string& filename) {
        std::ifstream file(filename);
        if(!file.is_open()){
            throw std::runtime_error("Cannot open config file: " + filename);
        }
        
        std::string current_section;
        std::string line;
        size_t line_num = 0;
        
        while(std::getline(file, line)){
            line_num++;
            
            // Remove comments
            size_t comment = line.find('#');
            if(comment != std::string::npos){
                line = line.substr(0, comment);
            }
            
            line = trim(line);
            if(line.empty()) continue;
            
            // Check for section header [type name]
            if(line[0] == '[' && line.back() == ']'){
                current_section = line.substr(1, line.length() - 2);
                current_section = trim(current_section);
                continue;
            }
            
            // Parse key = value
            size_t eq = line.find('=');
            if(eq != std::string::npos){
                std::string key = trim(line.substr(0, eq));
                std::string value = trim(line.substr(eq + 1));
                sections[current_section][key] = value;
            }
        }
        
        file.close();
    }
    
    bool has_section(const std::string& section) const {
        return sections.find(section) != sections.end();
    }
    
    std::string get_string(const std::string& section, const std::string& key, 
                          const std::string& default_val = "") const {
        auto sec_it = sections.find(section);
        if(sec_it == sections.end()) return default_val;
        
        auto key_it = sec_it->second.find(key);
        if(key_it == sec_it->second.end()) return default_val;
        
        return key_it->second;
    }
    
    double get_double(const std::string& section, const std::string& key, 
                     double default_val = 0.0) {
        std::string val = get_string(section, key);
        if(val.empty()) return default_val;
        return parse_expression(val);
    }
    
    bool get_bool(const std::string& section, const std::string& key, 
                 bool default_val = false) const {
        std::string val = get_string(section, key);
        if(val.empty()) return default_val;
        return (val == "true" || val == "True" || val == "1" || val == "yes");
    }
    
    std::vector<std::string> get_sections_by_type(const std::string& type) const {
        std::vector<std::string> result;
        for(const auto& pair : sections){
            if(pair.first.substr(0, type.length()) == type){
                result.push_back(pair.first);
            }
        }
        return result;
    }
    
    std::string extract_name(const std::string& section) const {
        size_t space = section.find(' ');
        if(space != std::string::npos){
            return trim(section.substr(space + 1));
        }
        return section;
    }
};

// Extended Cavity class with named entity support
class CavityConfig {
private:
    std::unordered_map<std::string, size_t> polariton_map;
    std::unordered_map<std::string, size_t> phonon_map;
    
public:
    void load_from_ini(Cavity& cavity, const std::string& filename) {
        ConfigParser parser;
        parser.load(filename);
        
        // Check for custom seed
        if(parser.has_section("global")){
            std::string seed_str = parser.get_string("global", "random_seed");
            if(seed_str != "auto" && !seed_str.empty()){
                parser.set_seed(std::stoul(seed_str));
            }
        }
        
        // Phase 1: Create polaritons
        auto polariton_sections = parser.get_sections_by_type("polariton");
        for(const auto& section : polariton_sections){
            std::string name = parser.extract_name(section);
            double gamma = parser.get_double(section, "gamma", 1.0);
            double U = parser.get_double(section, "U", 0.0);  // Using omega as U
            
            size_t id = cavity.polaritons.size();
            cavity.polaritons.push_back(new PolaritonMode(gamma, U));
            polariton_map[name] = id;
            
            // Set initial conditions
            double re = parser.get_double(section, "initial_real", 0.0);
            double im = parser.get_double(section, "initial_imag", 0.0);
            cavity.polaritons[id]->set_value({re, im});
        }
        
        // Phase 2: Create phonons
        auto phonon_sections = parser.get_sections_by_type("phonon");
        for(const auto& section : phonon_sections){
            std::string name = parser.extract_name(section);
            double omega = parser.get_double(section, "omega", 20.0);
            double gamma = parser.get_double(section, "gamma", 0.05);
            
            size_t id = cavity.phonons.size();
            cavity.phonons.push_back(new PhononMode(omega, gamma));
            phonon_map[name] = id;
            
            // Set initial conditions
            double x = parser.get_double(section, "initial_position", 0.0);
            double v = parser.get_double(section, "initial_velocity", 0.0);
            cavity.phonons[id]->set_position(x);
            cavity.phonons[id]->set_velocity(v);
        }
        
        // Phase 3: Create reservoirs
        auto reservoir_sections = parser.get_sections_by_type("reservoir");
        for(const auto& section : reservoir_sections){
            std::string target = parser.get_string(section, "target");
            
            if(polariton_map.find(target) == polariton_map.end()){
                throw std::runtime_error("Reservoir target not found: " + target);
            }
            
            size_t pol_id = polariton_map[target];
            double coupling = parser.get_double(section, "coupling", 1.0);
            double tau = parser.get_double(section, "tau", 1.0);
            double power = parser.get_double(section, "power", 0.0);
            double alpha = std::sqrt(parser.get_double(section, "alpha", 1.0));
            double initial_n = parser.get_double(section, "n0", 0.0);
            
            cavity.polaritons[pol_id]->add_reservoir(coupling, tau, power, alpha, initial_n);
        }
        
        // Phase 4: Create couplings
        auto coupling_sections = parser.get_sections_by_type("coupling");
        for(const auto& section : coupling_sections){
            std::string from = parser.get_string(section, "from");
            std::string to = parser.get_string(section, "to");
            std::string phonon = parser.get_string(section, "phonon");
            
            if(polariton_map.find(from) == polariton_map.end()){
                throw std::runtime_error("Coupling 'from' not found: " + from);
            }
            if(polariton_map.find(to) == polariton_map.end()){
                throw std::runtime_error("Coupling 'to' not found: " + to);
            }
            if(phonon_map.find(phonon) == phonon_map.end()){
                throw std::runtime_error("Coupling phonon not found: " + phonon);
            }
            
            size_t from_id = polariton_map[from];
            size_t to_id = polariton_map[to];
            size_t ph_id = phonon_map[phonon];
            
            double J = parser.get_double(section, "J", 0.0);
            double g = parser.get_double(section, "g", 1.0);
            double delta = parser.get_double(section, "delta", 0.0);
            bool above = parser.get_bool(section, "above", true);
            
            cavity.polaritons[from_id]->connect(cavity.polaritons[to_id], 
                                                cavity.phonons[ph_id],
                                                J, g, delta, above);
        }
        
        // Phase 5: Create phonon pairings
        auto pairing_sections = parser.get_sections_by_type("pairing");
        for(const auto& section : pairing_sections){
            std::string phonon = parser.get_string(section, "phonon");
            std::string sites_str = parser.get_string(section, "sites");
            
            if(phonon_map.find(phonon) == phonon_map.end()){
                throw std::runtime_error("Pairing phonon not found: " + phonon);
            }
            
            // Parse comma-separated sites
            size_t comma = sites_str.find(',');
            if(comma == std::string::npos){
                throw std::runtime_error("Pairing sites must be comma-separated");
            }
            
            std::string site1 = sites_str.substr(0, comma);
            std::string site2 = sites_str.substr(comma + 1);
            site1.erase(0, site1.find_first_not_of(" \t"));
            site2.erase(0, site2.find_first_not_of(" \t"));
            
            if(polariton_map.find(site1) == polariton_map.end()){
                throw std::runtime_error("Pairing site not found: " + site1);
            }
            if(polariton_map.find(site2) == polariton_map.end()){
                throw std::runtime_error("Pairing site not found: " + site2);
            }
            
            size_t ph_id = phonon_map[phonon];
            size_t p1_id = polariton_map[site1];
            size_t p2_id = polariton_map[site2];
            
            double g = parser.get_double(section, "g", 1.0);
            double delta = parser.get_double(section, "delta", 0.0);
            
            cavity.phonons[ph_id]->add_pairing({cavity.polaritons[p1_id], 
                                                cavity.polaritons[p2_id]},
                                               delta, g);
        }
        
        // Initialize cavity
        double t0 = parser.get_double("global", "time", 0.0);
        cavity.current_time = t0;
        cavity.dopri5_stepper = boost::numeric::odeint::make_controlled<error_stepper>(
            cavity.abs_stepper_tol, cavity.rel_stepper_tol);
        
        for(auto* p : cavity.polaritons) p->check();
        for(auto* ph : cavity.phonons) ph->check();
        
        size_t N = 2 * cavity.polaritons.size() + 2 * cavity.phonons.size();
        for(auto* p : cavity.polaritons){
            if(p->get_reservoir()) ++N;
        }
        
        cavity.system_dimension = N;
        cavity.current_state.set_size(cavity.system_dimension);
        cavity.pack_state();
        
        std::cout << "Successfully loaded INI config:\n"
                  << "  " << cavity.polaritons.size() << " polaritons\n"
                  << "  " << cavity.phonons.size() << " phonons\n"
                  << "  dimension = " << cavity.system_dimension << std::endl;
    }
    
    size_t get_polariton_id(const std::string& name) const {
        auto it = polariton_map.find(name);
        if(it == polariton_map.end()){
            throw std::runtime_error("Polariton not found: " + name);
        }
        return it->second;
    }
    
    size_t get_phonon_id(const std::string& name) const {
        auto it = phonon_map.find(name);
        if(it == phonon_map.end()){
            throw std::runtime_error("Phonon not found: " + name);
        }
        return it->second;
    }
};