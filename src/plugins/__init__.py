from src.plugins.base import BaseClientPlugin, BaseServerPlugin
from src.plugins.fedfed_plugin import FedFedClientPlugin, FedFedServerPlugin


PLUGIN_REGISTRY = {
    'fedfed_prototype': {
        'client': FedFedClientPlugin,
        'server': FedFedServerPlugin,
    },
}


def resolve_plugin_name(options):
    plugin_name = options.get('plugin_name', 'none')
    if plugin_name != 'none':
        return plugin_name
    if options.get('use_fedfed_plugin', False):
        return 'fedfed_prototype'
    return None


def build_client_plugin(options, model, gpu):
    plugin_name = resolve_plugin_name(options)
    if plugin_name is None:
        return None
    if plugin_name not in PLUGIN_REGISTRY:
        raise ValueError('Unsupported plugin: {}'.format(plugin_name))
    return PLUGIN_REGISTRY[plugin_name]['client'](options, model, gpu)


def build_server_plugin(options, gpu):
    plugin_name = resolve_plugin_name(options)
    if plugin_name is None:
        return None
    if plugin_name not in PLUGIN_REGISTRY:
        raise ValueError('Unsupported plugin: {}'.format(plugin_name))
    return PLUGIN_REGISTRY[plugin_name]['server'](options, gpu)


__all__ = [
    'BaseClientPlugin',
    'BaseServerPlugin',
    'FedFedClientPlugin',
    'FedFedServerPlugin',
    'build_client_plugin',
    'build_server_plugin',
    'resolve_plugin_name',
]
