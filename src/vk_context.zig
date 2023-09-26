const std = @import("std");
const vk = @import("vk.zig");

pub const VkAssert = struct {
    pub fn basic(result: vk.Result) !void {
        switch (result) {
            .success => return,
            else => return error.Unknown,
        }
    }

    pub fn withMessage(result: vk.Result, message: []const u8) !void {
        switch (result) {
            .success => return,
            else => {
                std.log.err("{s} {s}", .{ @tagName(result), message });
                return error.Unknown;
            },
        }
    }
};

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
    .destroySurfaceKHR = true,
    .enumeratePhysicalDevices = true,
    .enumerateDeviceExtensionProperties = true,
    .getPhysicalDeviceProperties = true,
    .getPhysicalDeviceFeatures = true,
    .getPhysicalDeviceQueueFamilyProperties = true,
    .getPhysicalDeviceSurfaceSupportKHR = true,
    .getPhysicalDeviceSurfaceCapabilitiesKHR = true,
    .getPhysicalDeviceSurfaceFormatsKHR = true,
    .getPhysicalDeviceSurfacePresentModesKHR = true,
    .createDevice = true,
    .getDeviceProcAddr = true,
});

pub const DeviceDispatch = vk.DeviceWrapper(.{
    .destroyDevice = true,
    .getDeviceQueue = true,
    .createSwapchainKHR = true,
    .destroySwapchainKHR = true,
    .getSwapchainImagesKHR = true,
    .createImageView = true,
    .destroyImageView = true,
    .createShaderModule = true,
    .destroyShaderModule = true,
    .createRenderPass = true,
    .destroyRenderPass = true,
    .createPipelineLayout = true,
    .destroyPipelineLayout = true,
    .createGraphicsPipelines = true,
    .destroyPipeline = true,
    .createFramebuffer = true,
    .destroyFramebuffer = true,
    .createCommandPool = true,
    .destroyCommandPool = true,
    .allocateCommandBuffers = true,
    .freeCommandBuffers = true,
    .beginCommandBuffer = true,
    .resetCommandBuffer = true,
    .endCommandBuffer = true,
    .cmdBeginRenderPass = true,
    .cmdEndRenderPass = true,
    .cmdBindPipeline = true,
    .cmdSetViewport = true,
    .cmdSetScissor = true,
    .cmdDraw = true,
    .createSemaphore = true,
    .destroySemaphore = true,
    .createFence = true,
    .destroyFence = true,
    .waitForFences = true,
    .resetFences = true,
    .acquireNextImageKHR = true,
    .queueSubmit = true,
    .queuePresentKHR = true,
    .deviceWaitIdle = true,
});
