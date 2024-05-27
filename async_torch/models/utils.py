from async_torch.layers.compression import QuantizSimple
from async_torch.models.models_ResNet import make_layers_resnet18, make_layers_resnet34, make_layers_resnet50, \
    make_layers_resnet101_2
from async_torch.models.models_RevNet import make_layers_revnet18, make_layers_revnet34, \
    make_layers_revnet50_variant, make_layers_revnet50, make_layers_revnet101_variant, \
    make_layers_revnet101
from async_torch.models.models_VGG import make_layers_VGG


def get_model(dataset, model, last_bn_zero_init, store_input=True, store_param=True, store_vjp=False,
              quantizer=QuantizSimple, accumulation_steps=1, accumulation_averaging=False, approximate_input=False):
    if dataset in ['cifar10', 'cifar100']:
        num_classes = 10 if dataset == 'cifar10' else 100
    elif dataset in ['imagenet32', 'imagenet']:
        num_classes = 1000
    else:
        raise ValueError(f'Wrong dataset ({dataset}).')

    # layerwise vgg partitioning
    if model == 'vgg':
        assert dataset != 'imagenet', 'VGG architecture is not compatible with full size Imagenet.'
        arch = make_layers_VGG(nclass=num_classes, store_param=store_param, store_vjp=store_vjp, quantizer=quantizer,
                               accumulation_steps=accumulation_steps, accumulation_averaging=accumulation_averaging)

    # blockwise resnet partitioning
    elif model == 'resnet18':
        arch = make_layers_resnet18(dataset, nclass=num_classes, last_bn_zero_init=last_bn_zero_init,
                                    store_param=store_param, store_vjp=store_vjp,
                                    quantizer=quantizer, accumulation_steps=accumulation_steps,
                                    accumulation_averaging=accumulation_averaging)
    elif model == 'resnet34':
        arch = make_layers_resnet34(dataset, nclass=num_classes, store_param=store_param, store_vjp=store_vjp,
                                    quantizer=quantizer, accumulation_steps=accumulation_steps,
                                    accumulation_averaging=accumulation_averaging)
    elif model == 'resnet50':
        arch = make_layers_resnet50(dataset, nclass=num_classes, store_param=store_param, store_vjp=store_vjp,
                                    quantizer=quantizer, accumulation_steps=accumulation_steps,
                                    accumulation_averaging=accumulation_averaging)
    elif model == 'resnet101':
        arch = make_layers_resnet101_2(dataset, nclass=num_classes, store_param=store_param, store_vjp=store_vjp,
                                       quantizer=quantizer, accumulation_steps=accumulation_steps,
                                       accumulation_averaging=accumulation_averaging)

    # blockwise revnet partitioning
    elif model == 'revnet18':
        arch = make_layers_revnet18(dataset, nclass=num_classes, last_bn_zero_init=last_bn_zero_init,
                                    store_input=store_input, store_param=store_param, store_vjp=store_vjp,
                                    quantizer=quantizer, accumulation_steps=accumulation_steps,
                                    accumulation_averaging=accumulation_averaging,
                                    approximate_input=approximate_input)
    elif model == 'revnet34':
        arch = make_layers_revnet34(dataset, nclass=num_classes, last_bn_zero_init=last_bn_zero_init,
                                    store_input=store_input, store_param=store_param, store_vjp=store_vjp,
                                    quantizer=quantizer, accumulation_steps=accumulation_steps,
                                    accumulation_averaging=accumulation_averaging,
                                    approximate_input=approximate_input)
    elif model == 'revnet50':
        arch = make_layers_revnet50(dataset, nclass=num_classes, last_bn_zero_init=last_bn_zero_init,
                                    store_input=store_input, store_param=store_param, store_vjp=store_vjp,
                                    quantizer=quantizer, accumulation_steps=accumulation_steps,
                                    accumulation_averaging=accumulation_averaging,
                                    approximate_input=approximate_input)
    elif model == 'revnet50_variant':
        arch = make_layers_revnet50_variant(dataset, nclass=num_classes, last_bn_zero_init=last_bn_zero_init,
                                            store_input=store_input, store_param=store_param, store_vjp=store_vjp,
                                            quantizer=quantizer, accumulation_steps=accumulation_steps,
                                            accumulation_averaging=accumulation_averaging,
                                            approximate_input=approximate_input)
    elif model == 'revnet101':
        arch = make_layers_revnet101(dataset, nclass=num_classes, last_bn_zero_init=last_bn_zero_init,
                                     store_input=store_input, store_param=store_param, store_vjp=store_vjp,
                                     quantizer=quantizer, accumulation_steps=accumulation_steps,
                                     accumulation_averaging=accumulation_averaging,
                                     approximate_input=approximate_input)
    elif model == 'revnet101_variant':
        arch = make_layers_revnet101_variant(dataset, nclass=num_classes, last_bn_zero_init=last_bn_zero_init,
                                             store_input=store_input, store_param=store_param, store_vjp=store_vjp,
                                             quantizer=quantizer, accumulation_steps=accumulation_steps,
                                             accumulation_averaging=accumulation_averaging,
                                             approximate_input=approximate_input)
    else:
        raise ValueError(f'Wrong architecture ({model}).')
    return arch
