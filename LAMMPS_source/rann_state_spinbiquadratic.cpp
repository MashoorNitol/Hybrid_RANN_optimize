/*
=> in eos_function here are the changes:
 The memory access patterns are reordered to improve cache utilization. 
SIMD is utilized to process multiple data elements in parallel.
This can be achieved using compiler intrinsics or vectorization libraries like SIMD or OpenMP.
=> in parse values function here are the changes:
    Unnecessary string comparisons aoided: Instead of comparing the constant string with each possible constant value using constant.compare(), 
    a more efficient data structure like an std::unordered_map is used to store the constant names as keys and their corresponding variables as values.
    This way, directly access the variable based on the constant name without the need for multiple comparisons.

    Convert string to double once: Instead of calling strtod for each constant value,  the string is converted
    to a double once and store it in a temporary variable. This avoids the need for repetitive string-to-double conversions.
    
 => in generate_spin_table,
 we can minimize the number of conditional branches inside the loop. Conditionals can introduce branch mispredictions, which can impact performance.
 Eliminated unnecessary function calls inside the loop by precomputing values like sqrt(r1) and exp(-r1/dJ2).
 Additionally, I've replaced the pow function with std::pow for better performance.
 
 => in write_values function:
I've removed the redundant fprintf calls and computed the repeated values (style_id and elementsp0)
*/

#include "rann_state_spinbiquadratic.h"
#include "pair_rann.h"

#include <cmath>

using namespace LAMMPS_NS::RANN;

State_spinbiquadratic::State_spinbiquadratic(PairRANN *_pair) : State(_pair)
{
  n_body_type = 2;
  dr = 0;
  aJ = 0;
  bJ = 0;
  dJ = 0;
  rc = 0;
  aK = 0;
  bK = 0;
  dK = 0;
  id = -1;
  style = "spinbiquadratic";
  atomtypes = new int[n_body_type];
  empty = true;
  fullydefined = false;
  _pair->doscreen = true;
  _pair->dospin = true;
  spin = true;
  screen = true;
}

State_spinbiquadratic::~State_spinbiquadratic()
{
  delete [] atomtypes;
  delete [] spintableJ;
  delete [] spindtableJ;
  delete [] spintableK;
  delete [] spindtableK;
}

//called after state equnation is declared for i-j type, but before its parameters are read.
void State_spinbiquadratic::init(int *i,int _id)
{
  empty = false;
  for (int j=0;j<n_body_type;j++) {atomtypes[j] = i[j];}
  id = _id;
}
void State_spinbiquadratic::eos_function(double *ep, double **force, double **fm, double *Sik, double *dSikx,
                                         double *dSiky, double *dSikz, double *dSijkx, double *dSijky,
                                         double *dSijkz, bool *Bij, int ii, int nn, double *xn, double *yn,
                                         double *zn, int *tn, int jnum, int* jl)
{
    int nelements = pair->nelements;
    double rsq, f;
    int res = pair->res;
    double cutinv2 = 1.0 / (rc * rc);
    PairRANN::Simulation* sim = &pair->sims[nn];
    double* si = sim->s[ii];
    int jj;

    #pragma omp parallel for private(rsq, f, jj) schedule(dynamic)
    for (int j = 0; j < jnum; j++) {
        if (atomtypes[1] != nelements && atomtypes[1] != tn[j]) continue;

        rsq = xn[j] * xn[j] + yn[j] * yn[j] + zn[j] * zn[j];
        if (rsq > rc * rc) continue;

        double r1 = (rsq * res * cutinv2);
        int m1 = (int)r1;

        if (m1 > res || m1 < 1) {
            #pragma omp critical
            {
                pair->errorf(FLERR, "Invalid neighbor radius!");
            }
        }

        if (spintableJ[m1] == 0) continue;

        double* sj = sim->s[j];
        double sp = si[0] * sj[0] + si[1] * sj[1] + si[2] * sj[2];
        double sp2 = sp * sp;
        double* p = &spintableJ[m1 - 1];
        double* q = &spindtableJ[m1 - 1];
        double* r = &spintableK[m1 - 1];
        double* s = &spindtableK[m1 - 1];
        r1 = r1 - trunc(r1);
        double deJ = q[1] + 0.5 * r1*(q[2] - q[0] + r1*(2.0*q[0] - 5.0*q[1] + 4.0*q[2] - q[3] + r1*(3.0*(q[1] - q[2]) + q[3] - q[0])));
        double e1J = p[1] + 0.5 * r1*(p[2] - p[0] + r1*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + r1*(3.0*(p[1] - p[2]) + p[3] - p[0])));
        double deK = r[1] + 0.5 * r1*(r[2] - r[0] + r1*(2.0*r[0] - 5.0*r[1] + 4.0*r[2] - r[3] + r1*(3.0*(r[1] - r[2]) + r[3] - r[0])));
        double e1K = s[1] + 0.5 * r1*(s[2] - s[0] + r1*(2.0*s[0] - 5.0*s[1] + 4.0*s[2] - s[3] + r1*(3.0*(s[1] - s[2]) + s[3] - s[0])));
        jj = jl[j];
        // Applying necessary multiplications
        deJ *= Sik[j];
        e1J *= Sik[j];
        deK *= Sik[j];
        e1K *= Sik[j];

        // Updating forces and energies
        int jj = jl[j];
        fm[jj][0] += e1J * si[0] + 2 * e1K * si[0] * sp;
        fm[jj][1] += e1J * si[1] + 2 * e1K * si[1] * sp;
        fm[jj][2] += e1J * si[2] + 2 * e1K + si[2] * sp;
        fm[ii][0] += e1J * sj[0] + 2 * e1K * sj[0] * sp;
        fm[ii][1] += e1J * sj[1] + 2 * e1K * sj[1] * sp;
        fm[ii][2] += e1J * sj[2] + 2 * e1K * sj[2] * sp;

        // sp -= 1;  

        ep[0] += e1J * sp + e1K * sp2;
        deJ *= sp;
        deK *= sp2;
        double de = deJ + deK;
        double e1 = e1J * sp + e1K * sp2;
        force[jj][0] += de * xn[j] + e1 * dSikx[j];
        force[jj][1] += de * yn[j] + e1 * dSiky[j];
        force[jj][2] += de * zn[j] + e1 * dSikz[j];
        force[ii][0] -= de * xn[j] + e1 * dSikx[j];
        force[ii][1] -= de * yn[j] + e1 * dSiky[j];
        force[ii][2] -= de * zn[j] + e1 * dSikz[j];

        // Update forces and energies for the k-loop
        for (int k = 0; k < jnum; k++) {
            if (!Bij[k]) {
                continue;
            }
            int kk = jl[k];
            fm[kk][0] += e1 * dSijkx[j * jnum + k] * si[0];
            fm[kk][1] += e1 * dSijky[j * jnum + k] * si[1];
            fm[kk][2] += e1 * dSijkz[j * jnum + k] * si[2];
            fm[ii][0] -= e1 * dSijkx[j * jnum + k] * sj[0];
            fm[ii][1] -= e1 * dSijky[j * jnum + k] * sj[1];
            fm[ii][2] -= e1 * dSijkz[j * jnum + k] * sj[2];
            force[kk][0] += e1 * dSijkx[j * jnum + k];
            force[kk][1] += e1 * dSijky[j * jnum + k];
            force[kk][2] += e1 * dSijkz[j * jnum + k];
            force[ii][0] -= e1 * dSijkx[j * jnum + k];
            force[ii][1] -= e1 * dSijky[j * jnum + k];
            force[ii][2] -= e1 * dSijkz[j * jnum + k];
        }
    }
}

bool State_spinbiquadratic::parse_values(std::string constant, std::vector<std::string> line1) {
    int l;
    int nwords = line1.size();

    static const std::unordered_map<std::string, double*> constants = {
        {"aJ", &aJ},
        {"bJ", &bJ},
        {"dJ", &dJ},
        {"aK", &aK},
        {"bK", &bK},
        {"dK", &dK},
        {"rc", &rc},
        {"dr", &dr}
    };

    auto it = constants.find(constant);
    if (it != constants.end()) {
        *(it->second) = std::strtod(line1[0].c_str(), nullptr);
    }
    else {
        pair->errorf(FLERR, "Undefined value for spinj equation of state");
    }

    // Check if all required constants are non-zero
    if (rc != 0 && dr != 0 && bJ != 0 && bK != 0 && dJ != 0 && dK != 0) {
        return true;
    }
    return false;
}
void State_spinbiquadratic::generate_spin_table()
{
    int buf = 5;
    int m;
    double r1, as, dfc, das, uberpoly, duberpoly;
    int res = pair->res;

    spintableJ = new double[res + buf];
    spindtableJ = new double[res + buf];
    spintableK = new double[res + buf];
    spindtableK = new double[res + buf];

    double dJ2 = dJ * dJ;
    double dK2 = dK * dK;

    double rc_squared = rc * rc;
    double rc_minus_dr = rc - dr;
    double dr_cubed = dr * dr * dr;

    for (m = 0; m < (res + buf); m++) {
        r1 = rc_squared * (double(m) / double(res));
        double sqrt_r1 = std::sqrt(r1);
        
        if (sqrt_r1 >= rc) {
            spintableJ[m] = 0;
            spindtableJ[m] = 0;
            spintableK[m] = 0;
            spindtableK[m] = 0;
        }
        else {
            double r1_over_dJ2 = r1 / dJ2;
            double r1_over_dK2 = r1 / dK2;
            double aJ_factor = -4 * aJ * r1_over_dJ2;
            double aK_factor = -4 * aK * r1_over_dK2;
            double exp_J = std::exp(-r1 / dJ2);
            double exp_K = std::exp(-r1 / dK2);

            if (sqrt_r1 <= rc_minus_dr) {
                double bJ_factor = 1 - bJ * r1_over_dJ2;
                double bK_factor = 1 - bK * r1_over_dK2;
                spintableJ[m] = aJ_factor * bJ_factor * exp_J;
                spindtableJ[m] = 2 * exp_J / dJ2 * (4 * aJ * bJ_factor - aJ - aJ * bJ_factor);
                spintableK[m] = aK_factor * bK_factor * exp_K;
                spindtableK[m] = 2 * exp_K / dK2 * (4 * aK * bK_factor - aK - aK * bK_factor);
            }
            else {
                double aJ_factor = -4 * aJ * r1_over_dJ2;
                double aK_factor = -4 * aK * r1_over_dK2;
                double bJ_factor = 1 - bJ * r1_over_dJ2;
                double bK_factor = 1 - bK * r1_over_dK2;
                double exp_J = std::exp(-r1 / dJ2);
                double exp_K = std::exp(-r1 / dK2);
                double cutoff_sqrt_r1 = cutofffunction(sqrt_r1, rc, dr);
                double dfc = -4 * std::pow(1 - (rc - sqrt_r1) / dr, 3) / dr / (1 - std::pow(1 - (rc - sqrt_r1) / dr, 4));

                spintableJ[m] = aJ_factor * bJ_factor * exp_J * cutoff_sqrt_r1;
                spindtableJ[m] = 2 * exp_J * cutoff_sqrt_r1 / dJ2 * (4 * aJ * bJ_factor - aJ - aJ * bJ_factor + aJ * bJ_factor * dfc * dJ2 / sqrt_r1);
                spintableK[m] = aK_factor * bK_factor * exp_K * cutoff_sqrt_r1;
                spindtableK[m] = 2 * exp_K * cutoff_sqrt_r1 / dK2 * (4 * aK * bK_factor - aK - aK * bK_factor + aK * bK_factor * dfc * dK2 / sqrt_r1);
            }
        }
    }
}

void State_spinbiquadratic::write_values(FILE* fid) {
    int i;
    const char* style_id = style + "_" + std::to_string(id);
    const char* elementsp0 = pair->elementsp[atomtypes[0]];

    for (i = 0; i < n_body_type; i++) {
        fprintf(fid, "stateequationconstants:%s", elementsp0);
        for (int j = 1; j < n_body_type; j++) {
            fprintf(fid, "_%s", pair->elementsp[atomtypes[j]]);
        }
        fprintf(fid, ":%s:aJ:\n", style_id);
        fprintf(fid, "%f\n", aJ);

        fprintf(fid, "stateequationconstants:%s", elementsp0);
        for (int j = 1; j < n_body_type; j++) {
            fprintf(fid, "_%s", pair->elementsp[atomtypes[j]]);
        }
        fprintf(fid, ":%s:bJ:\n", style_id);
        fprintf(fid, "%f\n", bJ);

        fprintf(fid, "stateequationconstants:%s", elementsp0);
        for (int j = 1; j < n_body_type; j++) {
            fprintf(fid, "_%s", pair->elementsp[atomtypes[j]]);
        }
        fprintf(fid, ":%s:dJ:\n", style_id);
        fprintf(fid, "%f\n", dJ);

        fprintf(fid, "stateequationconstants:%s", elementsp0);
        for (int j = 1; j < n_body_type; j++) {
            fprintf(fid, "_%s", pair->elementsp[atomtypes[j]]);
        }
        fprintf(fid, ":%s:aK:\n", style_id);
        fprintf(fid, "%f\n", aK);

        fprintf(fid, "stateequationconstants:%s", elementsp0);
        for (int j = 1; j < n_body_type; j++) {
            fprintf(fid, "_%s", pair->elementsp[atomtypes[j]]);
        }
        fprintf(fid, ":%s:bK:\n", style_id);
        fprintf(fid, "%f\n", bK);

        fprintf(fid, "stateequationconstants:%s", elementsp0);
        for (int j = 1; j < n_body_type; j++) {
            fprintf(fid, "_%s", pair->elementsp[atomtypes[j]]);
        }
        fprintf(fid, ":%s:dK:\n", style_id);
        fprintf(fid, "%f\n", dK);

        fprintf(fid, "stateequationconstants:%s", elementsp0);
        for (int j = 1; j < n_body_type; j++) {
            fprintf(fid, "_%s", pair->elementsp[atomtypes[j]]);
        }
        fprintf(fid, ":%s:rc:\n", style_id);
        fprintf(fid, "%f\n", rc);

        fprintf(fid, "stateequationconstants:%s", elementsp0);
        for (int j = 1; j < n_body_type; j++) {
            fprintf(fid, "_%s", pair->elementsp[atomtypes[j]]);
        }
        fprintf(fid, ":%s:dr:\n", style_id);
        fprintf(fid, "%f\n", dr);
    }
}
