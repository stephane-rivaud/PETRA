from async_torch.layers.compression import QuantizSimple
from async_torch.models.models_ResNet import make_layers_resnet18, make_layers_resnet34, make_layers_resnet50, \
    make_layers_resnet101
from async_torch.models.models_RevNet import make_layers_revnet18, make_layers_revnet34, \
    make_layers_revnet50_variant, make_layers_revnet50, make_layers_revnet101_variant, \
    make_layers_revnet101
from async_torch.models.models_VGG import make_layers_VGG


def get_model(dataset,
              model,
              last_bn_zero_init,
              store_vjp=False,
              store_input=True,
              store_param=True,
              approximate_input=False,
              accumulation_steps=1,
              accumulation_averaging=False,
              quantizer=QuantizSimple,
              quantize_forward_communication=False,
              quantize_backward_communication=False,
              quantize_buffer=False,
              ):
    if dataset in ['cifar10', 'cifar100']:
        num_classes = 10 if dataset == 'cifar10' else 100
    elif dataset in ['imagenet32', 'imagenet']:
        num_classes = 1000
    else:
        raise ValueError(f'Wrong dataset ({dataset}).')

    # layerwise vgg partitioning
    if model == 'vgg':
        assert dataset != 'imagenet', 'VGG architecture is not compatible with full size Imagenet.'
        arch = make_layers_VGG(nclass=num_classes,
                               store_vjp=store_vjp,
                               store_param=store_param,
                               accumulation_steps=accumulation_steps,
                               accumulation_averaging=accumulation_averaging,
                               quantizer=quantizer,
                               quantize_forward_communication=quantize_forward_communication,
                               quantize_backward_communication=quantize_backward_communication,
                               quantize_buffer=quantize_buffer,
                               )

    # blockwise resnet partitioning
    elif 'resnet' in model:
        match model:
            case 'resnet18':
                model_constructor = make_layers_resnet18
            case 'resnet34':
                model_constructor = make_layers_resnet34
            case 'resnet50':
                model_constructor = make_layers_resnet50
            case 'resnet101':
                model_constructor = make_layers_resnet101
            case _:
                raise ValueError(f'Wrong architecture ({model}).')
        arch = model_constructor(dataset,
                                 nclass=num_classes,
                                 last_bn_zero_init=last_bn_zero_init,
                                 store_vjp=store_vjp,
                                 store_param=store_param,
                                 accumulation_steps=accumulation_steps,
                                 accumulation_averaging=accumulation_averaging,
                                 quantizer=quantizer,
                                 quantize_forward_communication=quantize_forward_communication,
                                 quantize_backward_communication=quantize_backward_communication,
                                 quantize_buffer=quantize_buffer,
                                 )

    # blockwise revnet partitioning
    elif 'revnet' in model:
        match model:
            case 'revnet18':
                model_constructor = make_layers_revnet18
            case 'revnet34':
                model_constructor = make_layers_revnet34
            case 'revnet50':
                model_constructor = make_layers_revnet50
            case 'revnet50_variant':
                model_constructor = make_layers_revnet50_variant
            case 'revnet101':
                model_constructor = make_layers_revnet101
            case _:
                raise ValueError(f'Wrong architecture ({model}).')

        arch = model_constructor(dataset,
                                 nclass=num_classes,
                                 last_bn_zero_init=last_bn_zero_init,
                                 store_vjp=store_vjp,
                                 store_param=store_param,
                                 store_input=store_input,
                                 approximate_input=approximate_input,
                                 accumulation_steps=accumulation_steps,
                                 accumulation_averaging=accumulation_averaging,
                                 quantizer=quantizer,
                                 quantize_forward_communication=quantize_forward_communication,
                                 quantize_backward_communication=quantize_backward_communication,
                                 quantize_buffer=quantize_buffer,
                                 )

    else:
        raise ValueError(f'Wrong architecture ({model}).')
    return arch
