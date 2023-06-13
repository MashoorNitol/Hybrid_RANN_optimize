#ifndef LMP_RANN_STATE_SPINBIQUADRATIC_H
#define LMP_RANN_STATE_SPINBIQUADRATIC_H

#include "rann_stateequation.h"

namespace LAMMPS_NS {
namespace RANN {

  class State_spinbiquadratic : public State {
   public:
    State_spinbiquadratic(class PairRANN *);
    ~State_spinbiquadratic();
    void eos_function(double*,double**,double**,double*,double*,double*,double*,double*,double*,
                                  double*,bool*,int,int,double*,double*,double*,int*,int,int*);//screen,spin
    bool parse_values(std::string, std::vector<std::string>);
    void generate_spin_table();
    void allocate(){generate_spin_table();}
    void write_values(FILE *); 
    void init(int*,int);
    double aJ;
    double bJ;
    double dJ;
    double aK;
    double bK;
    double dK;
    double dr;
    double *spintableJ;
    double *spindtableJ;
    double *spintableK;
    double *spindtableK;
  };



}    // namespace RANN
}    // namespace LAMMPS_NS

#endif /* LMP_RANN_STATE_ROSE_H_ */
