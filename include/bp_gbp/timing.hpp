#ifndef TIMING_HPP_
#define TIMING_HPP_

#include <chrono> // for std::chrono functions


class Timer {
        private:
                // Type aliases to make accessing nested type easier
                using clock_t = std::chrono::steady_clock;
                using second_t = std::chrono::duration<double, std::ratio<1> >;
                
                std::chrono::time_point<clock_t> m_beg;
        public:
                Timer();
                void reset();
                double elapsed() const;
};

#endif