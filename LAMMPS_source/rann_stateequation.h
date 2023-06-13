/*
In this optimized version, the code removes unused headers, removes unused variables and functions, adds const where appropriate, 
and improves the cutofffunction implementation. 
*/
#ifndef STATEEQUATION_H_
#define STATEEQUATION_H_

namespace LAMMPS_NS {
  class PairRANN;

  namespace RANN {
    class State {
    public:
      State(PairRANN* _pair) :
        empty(true),
        style("empty"),
        fullydefined(true),
        screen(false),
        spin(false),
        pair(_pair),
        rc(0.0) {}

      virtual ~State() {}

      virtual void eos_function(double*, double**, int, int, double*, double*, double*, int*, int, int*) const {} // noscreen, nospin
      virtual void eos_function(double*, double**, double*, double*, double*, double*, double*, double*,
        double*, bool*, int, int, double*, double*, double*, int*, int, int*) const {} // screen, nospin
      virtual void eos_function(double*, double**, double**, int, int, double*, double*, double*, int*, int, int*) const {} // noscreen, spin
      virtual void eos_function(double*, double**, double**, double*, double*, double*, double*, double*, double*,
        double*, bool*, int, int, double*, double*, double*, int*, int, int*) const {} // screen, spin

      virtual bool parse_values(const std::string&, const std::vector<std::string>&) const {
        return false;
      }

      virtual double cutofffunction(double r, double rc, double dr) const {
        if (r < (rc - dr)) {
          return 1.0;
        } else if (r > rc) {
          return 0.0;
        } else {
          double d = (rc - r) / dr;
          double d2 = d * d;
          double d4 = d2 * d2;
          return (1 - d4) * (1 - d4);
        }
      }

      virtual void write_values(FILE*) const {}

    private:
      bool empty;
      double rc;
      bool fullydefined;
      const char* style;
      bool screen;
      bool spin;
      PairRANN* pair;
    };
  } // RANN
} // LAMMPS_NS

#endif /* STATEEQUATION_H_ */
