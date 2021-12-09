/*
 * dropout.cpp
 *
 *  Created on: Nov 26, 2021
 *      Author: Maciej Kozarzewski
 */

#include <avocado/reference_backend.h>
#include <avocado/backend/backend_descriptors.hpp>
#include "utils.hpp"

namespace
{

}

namespace avocado
{
	namespace backend
	{
		avStatus_t refDropoutForward(avContextDescriptor_t context, const avDropoutDescriptor_t config, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem, avMemoryDescriptor_t states)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}
		avStatus_t refDropoutBackward(avContextDescriptor_t context, const avDropoutDescriptor_t config, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem, const avTensorDescriptor_t states)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}
	} /* namespace backend */
} /* namespace avocado */

