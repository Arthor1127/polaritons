#include "polariton.h"
#include <fstream>
#include <random>
#include <cstdio>
#include <vector>

std::mt19937 rng{std::random_device{}()};
std::uniform_real_distribution<double> uniform(0.0, 1.0);

/* ================= Live plot helper ================= */
struct LivePlot {
    FILE* gp;

    LivePlot() {
        gp = popen("gnuplot -persistent", "w");
        fprintf(gp, "set multiplot layout 3,1 title 'Polaritons + Phonon'\n");
        fprintf(gp, "set grid\n");
        fflush(gp);
    }

    void update(const std::vector<double>& t,
                const std::vector<double>& re1,
                const std::vector<double>& im1,
                const std::vector<double>& re2,
                const std::vector<double>& im2,
                const std::vector<double>& x)
    {
        auto send = [&](const std::vector<double>& y) {
            for (size_t i = 0; i < t.size(); ++i)
                fprintf(gp, "%lf %lf\n", t[i], y[i]);
            fprintf(gp, "e\n");
        };

        /* -------- Polariton 1 -------- */
        fprintf(gp, "set title 'Polariton 1'\n");
        fprintf(gp, "plot '-' w l title 'Re(ψ1)', '-' w l title 'Im(ψ1)'\n");
        send(re1);
        send(im1);

        /* -------- Polariton 2 -------- */
        fprintf(gp, "set title 'Polariton 2'\n");
        fprintf(gp, "plot '-' w l title 'Re(ψ2)', '-' w l title 'Im(ψ2)'\n");
        send(re2);
        send(im2);

        /* -------- Phonon -------- */
        fprintf(gp, "set title 'Phonon position'\n");
        fprintf(gp, "plot '-' w l title 'x'\n");
        send(x);

        fflush(gp);
    }

    ~LivePlot() {
        fprintf(gp, "unset multiplot\n");
        fflush(gp);
        pclose(gp);
    }
};


/* ======================= main ======================= */

int main(int argc, char* argv[]){
    if(argc != 4){
        std::cout << "Usage: ./program <steps> <index> <output_file>\n";
        return 1;
    }

    size_t steps     = std::stoi(argv[1]);
    size_t job_index = std::stoi(argv[2]);
    auto   file_name = argv[3];

    double driving_start = 0.0;
    double driving_stop  = 14.0;
    double driving_value =
        arma::linspace(driving_start, driving_stop, steps)[job_index];

    /* ============ Cavity Setup ============ */

    PolaritonMode site_1(1.0, 0.0);
    PolaritonMode site_2(1.0, 0.0);

    site_1.set_value({uniform(rng), uniform(rng)});
    site_2.set_value({uniform(rng), uniform(rng)});

    site_1.set_driving(0.0, 0.0);
    site_2.set_driving(0.0, 0.0);

    PhononMode phonon(20.0, 0.05);
    phonon.set_position(50.0 * uniform(rng));
    phonon.set_velocity(200.0 * uniform(rng));

    site_1.connect(&site_2, &phonon, 10.0, 1.0, 0.0, true);
    site_2.connect(&site_1, &phonon, 10.0, 1.0, 0.0, false);

    phonon.add_pairing({&site_1, &site_2}, 0.0, 1.0);

    std::vector<PolaritonMode*> sites   = {&site_1, &site_2};
    std::vector<PhononMode*>    phonons = {&phonon};

    Cavity model(sites, phonons, 0.0);

    /* ============ Integrator setup ============ */

    size_t transient  = 5e5;
    size_t stationary = 1e3;
    double delta_t    = 0.005;

    /* ============ Transient ============ */

    for(size_t i = 0; i < transient; ++i)
        model.do_step(delta_t);

    /* ============ Live plotting + averages ============ */

    LivePlot plot;

    std::vector<double> tbuf, re1, im1, re2, im2, xbuf;
    std::vector<double> avg{0.0, 0.0, 0.0};

    arma::vec aux;
    double t = 0.0;
    size_t plot_stride = 5;

    for(size_t i = 0; i < stationary; ++i){
        aux = model.get_state();

        double psi1_re = aux(0);
        double psi1_im = aux(1);
        double psi2_re = aux(2);
        double psi2_im = aux(3);
        double x       = aux(4);

        avg[0] += psi1_re*psi1_re + psi1_im*psi1_im;
        avg[1] += psi2_re*psi2_re + psi2_im*psi2_im;
        avg[2] += x*x;

        if(i % plot_stride == 0){
            tbuf.push_back(t);
            re1.push_back(psi1_re);
            im1.push_back(psi1_im);
            re2.push_back(psi2_re);
            im2.push_back(psi2_im);
            xbuf.push_back(x);

            plot.update(tbuf, re1, im1, re2, im2, xbuf);
        }

        model.do_step(delta_t);
        t += delta_t;
    }

    /* ============ Output ============ */

    std::ofstream output_file(file_name);
    output_file << driving_value << "\t";
    for(auto& e : avg)
        output_file << e / static_cast<double>(stationary) << "\t";
    output_file << "\n";
    output_file.close();

    return 0;
}
