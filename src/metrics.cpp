/*
 * metrics.cpp
 *
 *  Created on: Nov 25, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/reference_backend.h>

#include <Avocado/backend_descriptors.hpp>
#include "utils.hpp"

namespace
{
}

namespace avocado
{
	namespace backend
	{
		avStatus_t refMetricFunction(avContextDescriptor_t context, avMetricType_t metricType, const avTensorDescriptor_t outputDesc,
				const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc, const avMemoryDescriptor_t targetMem, void *result)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}
	} /* namespace backend */
} /* namespace avocado */

