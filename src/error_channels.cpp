#include <bp_gbp/error_channels.hpp>

NoisyChannel::NoisyChannel() :rd(), random_generator(rd())
{
  xt::random::seed(time(NULL));
}

void NoisyChannel::send_through_pauli_channel(xt::xarray<int> *y, double p_error, int type)
{
  if (type == 0)
  {
    depolarizing_channel(y,p_error);
  }
  if (type == 1)
  {
    xz_channel(y,p_error);
  }
  if (type == 2)
  {
    x_channel(y,p_error);
  }
}

void NoisyChannel::send_through_biased_channel(xt::xarray<int> *y,xt::xarray<double> p_initial)
{
  std::uniform_real_distribution<double> error_distribution(0,1);

  double p_x = p_initial(1);
  double p_z = p_initial(2);
  double p_y = p_initial(3);

  for (size_t i = 0; i < y->size(); i++)
  {
    double random_number = error_distribution(random_generator);
    if (random_number < p_x)
    {
      y->at(i) ^= 1;
    }
    else if ((p_x < random_number) && (random_number < p_x + p_z))
    {
      y->at(i) ^= 2;
    }
    else if ((p_x+p_z < random_number) && (random_number < p_x + p_z + p_y))
    {
      y->at(i) ^= 3;
    }
  }
}

void NoisyChannel::depolarizing_channel(xt::xarray<int> *y, double p_error)
{
  std::uniform_real_distribution<double> error_distribution(0,1); // for whether error happens or not
  std::uniform_int_distribution<> pauli(1, 3);            // for which type of error

  for (size_t i = 0; i < y->size(); i++)
  {
    double random_number = error_distribution(random_generator);
    if (random_number < p_error)
    {
      int random_pauli = pauli(random_generator);
      y->at(i) ^= random_pauli;
    }
  }
}

void depolarizing_channel(xt::xarray<int> *y, double p_error, std::mt19937 *random_generator)
{
  std::uniform_real_distribution<double> error_distribution(0,1); // for whether error happens or not
  std::uniform_int_distribution<> pauli(1, 3);            // for which type of error

  for (size_t i = 0; i < y->size(); i++)
  {
    double random_number = error_distribution(*random_generator);
    if (random_number < p_error)
    {
      int random_pauli = pauli(*random_generator);
      y->at(i) ^= random_pauli;
    }
  }
}

void NoisyChannel::xz_channel(xt::xarray<int> *y, double p_error)
{

  std::uniform_real_distribution<double> error_distribution(0,1);

  double p_xz = 1 - sqrt(1-p_error);

  for (size_t i = 0; i < y->size(); i++)
  {
    double random_number = error_distribution(random_generator);
    if (random_number < p_xz)
    {
      y->at(i) ^= 1;
    }
  }
  for (size_t i = 0; i < y->size(); i++)
  {
    double random_number = error_distribution(random_generator);
    if (random_number < p_xz)
    {
      y->at(i) ^= 2;
    }
  }
}


void xz_channel(xt::xarray<int> *y, double p_error, std::mt19937 *random_generator)
{
  std::uniform_int_distribution<> pauli(1, 2);            // for which type of error

  std::uniform_real_distribution<double> error_distribution(0,1);

  double p_xz = 1 - sqrt(1-p_error);

  for (size_t i = 0; i < y->size(); i++)
  {
    double random_number = error_distribution(*random_generator);
    if (random_number < p_xz)
    {
      y->at(i) ^= 1;
    }
    random_number = error_distribution(*random_generator);
    if (random_number < p_xz)
    {
      y->at(i) ^= 2;
    }
  }
}


void NoisyChannel::x_channel(xt::xarray<int> *y, double p_error)
{
  std::uniform_real_distribution<double> error_distribution(0,1);

  for (size_t i = 0; i < y->size(); i++)
  {
    double random_number = error_distribution(random_generator);
    if (random_number < p_error)
    {
      y->at(i) ^= 1;
    }
  }

}


void x_channel(xt::xarray<int> *y, double p_error, std::mt19937 *random_generator)
{
  std::uniform_real_distribution<double> error_distribution(0,1);

  for (size_t i = 0; i < y->size(); i++)
  {
    double random_number = error_distribution(*random_generator);
    if (random_number < p_error)
    {
      
      y->at(i) ^= 1;
    }
  }

}

void NoisyChannel::erasure_channel(xt::xarray<int> *y,xt::xarray<int> *erasures, double p_error)
{
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
      y->at(i) = random_pauli;
      erasures->at(i) = 1;
    }
  }
}

void NoisyChannel::biased_erasure_channel(xt::xarray<int> *y,xt::xarray<int> *erasures, xt::xarray<double> p_initial)
{
  std::uniform_real_distribution<double> error_distribution(0,1);  // for whether error happens or not

  double p_x = p_initial(1);
  double p_z = p_initial(2);
  double p_y = p_initial(3);

  for (size_t i = 0; i < y->size(); i++)
  {
    double random_number = error_distribution(random_generator);
    if (random_number < p_x)
    {
      y->at(i) ^= 1;
      erasures->at(i) = 1;
    }
    else if ((p_x < random_number) && (random_number < p_x + p_z))
    {
      y->at(i) ^= 2;
      erasures->at(i) = 1;
    }
    else if ((p_x+p_z < random_number) && (random_number < p_x + p_z + p_y))
    {
      y->at(i) ^= 3;
      erasures->at(i) = 1;
    }
  }
}

void erasure_channel(xt::xarray<int> *y,xt::xarray<int> *erasures, double p_error, std::mt19937 *random_generator)
{
  std::uniform_real_distribution<double> error_distribution(0,1);  // for whether error happens or not
  std::uniform_int_distribution<> pauli(1, 3);            // for which type of error

  for (size_t i = 0; i < y->size(); i++)
  {
    double random_number = error_distribution(*random_generator);
    // std::cout << "random number: " << random_number << " threshold: " << threshold << " p: " << p << std::endl;
    if (random_number < p_error)
    {
      int random_pauli = pauli(*random_generator);
      // std::cout << "random pauli: " << random_pauli << std::endl;
      y->at(i) = random_pauli;
      (*erasures)[i] = 1;
    }
  }
}

void NoisyChannel::const_weight_error_channel(xt::xarray<int> *y, int weight)
{
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

void const_weight_error_channel(xt::xarray<int> *y, int weight, std::mt19937 *random_generator)
{
  std::uniform_real_distribution<double> error_distribution(0,1); // for whether error happens or not
  std::uniform_int_distribution<> pauli(1, 3);            // for which type of error

  xt::xarray<int> qubit_positions = xt::arange(y->size());
  xt::xarray<int> error_positions = xt::random::choice(qubit_positions,weight,false);


  for (size_t i = 0; i < weight; i++)
  {
      int random_pauli = pauli(*random_generator);
      y->at(error_positions(i)) ^= random_pauli;
  }
}


void NoisyChannel::const_weight_error_channel(xt::xarray<int> *y, int weight, int max_qubit, int pauli)
{
  std::uniform_real_distribution<double> error_distribution(0,1); // for whether error happens or not
  // std::uniform_int_distribution<> pauli(1, n_paulis);            // for which type of error

  xt::xarray<int> qubit_positions = xt::arange<int>(max_qubit);
  xt::xarray<int> error_positions = xt::random::choice(qubit_positions,weight,false);

  // std::cout << "qubit_positions = " << qubit_positions << "\n";
  // std::cout << "error_positions = " << error_positions << "\n";
  

  for (size_t i = 0; i < weight; i++)
  {
      // int random_pauli = pauli(random_generator);
      y->at(error_positions(i)) ^= pauli;
  }
}

void NoisyChannel::const_weight_error_channel_T(xt::xarray<int> *y, int weight, int base_qubit, int max_qubit, int pauli)
{
  // std::uniform_real_distribution<double> error_distribution(0,1); // for whether error happens or not
  // std::uniform_int_distribution<> pauli(1, n_paulis);            // for which type of error

  xt::xarray<int> blocks = xt::arange<int>(max_qubit);
  xt::xarray<int> block_positions = xt::random::choice(blocks,5,false);


  // std::cout << "qubit_positions = " << qubit_positions << "\n";
  // std::cout << "error_positions = " << error_positions << "\n";
  
  for (auto b:block_positions)
  {
      xt::xarray<int> qubit_positions = xt::arange<int>(b*max_qubit,(b+1)*max_qubit);
      xt::xarray<int> error_positions = xt::random::choice(qubit_positions,weight,false);
      for (size_t i = 0; i < weight; i++)
      {
          // int random_pauli = pauli(random_generator);
          y->at(base_qubit+error_positions(i)) ^= pauli;
      }
  }

  
}

void const_weight_error_channel(xt::xarray<int> *y, int weight, int max_qubit, int n_paulis, std::mt19937 *random_generator)
{
  std::uniform_real_distribution<double> error_distribution(0,1); // for whether error happens or not
  std::uniform_int_distribution<> pauli(1, n_paulis);            // for which type of error

  xt::xarray<int> qubit_positions = xt::arange(max_qubit);
  xt::xarray<int> error_positions = xt::random::choice(qubit_positions,weight,false);


  for (size_t i = 0; i < weight; i++)
  {
      int random_pauli = pauli(*random_generator);
      y->at(error_positions(i)) ^= random_pauli;
  }
}
