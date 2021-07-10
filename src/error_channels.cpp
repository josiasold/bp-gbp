#include <bp_gbp/error_channels.hpp>

void depolarizing_channel(xt::xarray<int> *y, double p_error)
{
   /* initialize random seed: */
  std::random_device rd;                                  // obtain a random number from hardware
  std::mt19937 random_generator(rd());                                 // seed the generator
  std::uniform_real_distribution<double> error_distribution(0,1); // for whether error happens or not
  std::uniform_int_distribution<> pauli(1, 3);            // for which type of error

  for (size_t i = 0; i < y->size(); i++)
  {
    double random_number = error_distribution(random_generator);
    if (random_number < p_error)
    {
      int random_pauli = pauli(random_generator);
      (*y)[i] ^= random_pauli;
    }
  }
}

void xz_channel(xt::xarray<int> *y, double p_error)
{
  /* initialize random seed: */
  std::random_device rd;                                  // obtain a random number from hardware
  std::mt19937 random_generator(rd());                                 // seed the generator
  std::uniform_int_distribution<> pauli(1, 2);            // for which type of error

  std::uniform_real_distribution<double> error_distribution(0,1);

  for (size_t i = 0; i < y->size(); i++)
  {
    double random_number = error_distribution(random_generator);
    if (random_number < p_error)
    {
      int random_pauli = pauli(random_generator);
      (*y)[i] ^= random_pauli;
    }
  }
}


void x_channel(xt::xarray<int> *y, double p_error)
{
  std::random_device rd;                                  // obtain a random number from hardware
  std::mt19937 random_generator(rd());                                 // seed the generator
  std::uniform_real_distribution<double> error_distribution(0,1);

  for (size_t i = 0; i < y->size(); i++)
  {
    double random_number = error_distribution(random_generator);
    if (random_number < p_error)
    {
      
      (*y)[i] ^= 1;
    }
  }

}

void erasure_channel(xt::xarray<int> *y,xt::xarray<int> *erasures, double p_error)
{
  /* initialize random seed: */
  std::random_device rd;                                  // obtain a random number from hardware
  std::mt19937 random_generator(rd());                                 // seed the generator
  std::uniform_real_distribution<double> error_distribution(0,1);  // for whether error happens or not
  std::uniform_int_distribution<> pauli(1, 3);            // for which type of error

  for (size_t i = 0; i < y->size(); i++)
  {
    double random_number = error_distribution(random_generator);
    // std::cout << "random number: " << random_number << " threshold: " << threshold << " p: " << p << std::endl;
    if (random_number < p_error)
    {
      int random_pauli = pauli(random_generator);
      // std::cout << "random pauli: " << random_pauli << std::endl;
      (*y)[i] = random_pauli;
      (*erasures)[i] = 1;
    }
  }
}




void const_weight_error_channel(xt::xarray<int> *y, int weight)
{
   /* initialize random seed: */
  std::random_device rd;                                  // obtain a random number from hardware
  std::mt19937 random_generator(rd());                                 // seed the generator
  std::uniform_real_distribution<double> error_distribution(0,1); // for whether error happens or not
  std::uniform_int_distribution<> pauli(1, 3);            // for which type of error

  xt::xarray<int> qubit_positions = xt::arange(y->size());
  xt::xarray<int> error_positions = xt::random::choice(qubit_positions,weight,false);


  for (size_t i = 0; i < weight; i++)
  {
      int random_pauli = pauli(random_generator);
      y->at(error_positions(i)) ^= random_pauli;
  }
}

void const_weight_error_channel(xt::xarray<int> *y, int weight, int max_qubit, int n_paulis)
{
   /* initialize random seed: */
  std::random_device rd;                                  // obtain a random number from hardware
  std::mt19937 random_generator(rd());                                 // seed the generator
  std::uniform_real_distribution<double> error_distribution(0,1); // for whether error happens or not
  std::uniform_int_distribution<> pauli(1, n_paulis);            // for which type of error

  xt::xarray<int> qubit_positions = xt::arange(max_qubit);
  xt::xarray<int> error_positions = xt::random::choice(qubit_positions,weight,false);


  for (size_t i = 0; i < weight; i++)
  {
      int random_pauli = pauli(random_generator);
      y->at(error_positions(i)) ^= random_pauli;
  }
}
