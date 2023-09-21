const vk = @import("vk.zig");

pub const BaseDispatch = vk.BaseWrapper(.{
    .createInstance = true,
    .enumerateInstanceExtensionProperties = true,
    .getInstanceProcAddr = true,
});

pub const InstanceDispatch = vk.InstanceWrapper(.{});
