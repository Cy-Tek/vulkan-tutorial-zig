const vk = @import("vk.zig");

pub const BaseDispatch = vk.BaseWrapper(.{
    .createInstance = true,
    .enumerateInstanceExtensionProperties = true,
    .enumerateInstanceLayerProperties = true,
    .getInstanceProcAddr = true,
});

pub const InstanceDispatch = vk.InstanceWrapper(.{
    .destroyInstance = true,
    .createDebugUtilsMessengerEXT = true,
    .destroyDebugUtilsMessengerEXT = true,
});
