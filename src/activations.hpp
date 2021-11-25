/*
 * activations.hpp
 *
 *  Created on: Nov 11, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef ACTIVATIONS_HPP_
#define ACTIVATIONS_HPP_

#include <avocado/backend/backend_api.h>

#include "utils.hpp"

#include <algorithm>

namespace avocado
{
	namespace backend
	{
		template<typename T>
		T activation_forward(avActivationType_t type, T input) noexcept
		{
			switch (type)
			{
				case AVOCADO_ACTIVATION_LINEAR:
					return input;
				case AVOCADO_ACTIVATION_SIGMOID:
					return one<T>() / (one<T>() + exp(-input));
				case AVOCADO_ACTIVATION_TANH:
					return tanh(input);
				case AVOCADO_ACTIVATION_RELU:
					return std::max(zero<T>(), input);
				case AVOCADO_ACTIVATION_SELU:
					return static_cast<T>(1.05070098) * (input >= zero<T>() ? input : static_cast<T>(1.67326324) * expm1(input));
				case AVOCADO_ACTIVATION_ELU:
					return input >= zero<T>() ? input : expm1(input);
				case AVOCADO_ACTIVATION_EXPONENTIAL:
					return exp(input);
				case AVOCADO_ACTIVATION_SOFTPLUS:
					return log1p(exp(input));
				case AVOCADO_ACTIVATION_SOFTSIGN:
					return input / (fabs(input) + one<T>());
				default:
					return input;
			}
		}
		template<typename T>
		T activation_backward(avActivationType_t type, T gradient, T output) noexcept
		{
			switch (type)
			{
				case AVOCADO_ACTIVATION_LINEAR:
					return gradient;
				case AVOCADO_ACTIVATION_SIGMOID:
					return gradient * (one<T>() - output) * output;
				case AVOCADO_ACTIVATION_TANH:
					return gradient * (one<T>() - output) * (one<T>() + output);
				case AVOCADO_ACTIVATION_RELU:
					return output > zero<T>() ? gradient : zero<T>();
				case AVOCADO_ACTIVATION_SELU:
					return static_cast<T>(1.05070098) * gradient * (output >= zero<T>() ? one<T>() : static_cast<T>(1.67326324) * (output + one<T>()));
				case AVOCADO_ACTIVATION_ELU:
					return gradient * (output >= zero<T>() ? one<T>() : (output + one<T>()));
				case AVOCADO_ACTIVATION_EXPONENTIAL:
					return gradient * output;
				case AVOCADO_ACTIVATION_SOFTPLUS:
					return gradient * expm1(output) / exp(output);
				case AVOCADO_ACTIVATION_SOFTSIGN:
					return gradient / square(fabs(output / (one<T>() - sgn(output) * output)) + one<T>());
				case AVOCADO_ACTIVATION_SOFTMAX:
					return gradient * (one<T>() - output) * output;
				default:
					return gradient;
			}
		}
	} /* namespace backend */
} /* namespace avocado */

#endif /* ACTIVATIONS_HPP_ */
