// Minimal, synchronous Vulkan FP32 linear offload for inference only.
// The model's attention, normalization and scaling remain in official PyTorch.
#include <vulkan/vulkan.h>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <cstdio>
#include <string>
#include <cstdint>

static void check(VkResult r) {
    if(r!=VK_SUCCESS) throw std::runtime_error("Vulkan error "+std::to_string(r));
}
struct Buffer { VkBuffer b=VK_NULL_HANDLE; VkDeviceMemory mem=VK_NULL_HANDLE; void* map=nullptr; };
struct Layer { Buffer weights; VkDescriptorSet ds; uint32_t n,k; };
struct Engine {
    VkInstance instance=VK_NULL_HANDLE;
    VkPhysicalDevice physical=VK_NULL_HANDLE;
    VkDevice device=VK_NULL_HANDLE;
    VkQueue queue;
    VkCommandPool pool;
    VkCommandBuffer cmd;
    VkFence fence;
    VkDescriptorPool descriptorPool;
    VkDescriptorSetLayout dsLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;
    VkShaderModule shader;
    Buffer input,output,stageIn,stageOut;
    VkPhysicalDeviceMemoryProperties memories;
    std::vector<Layer> layers;
    const VkDeviceSize capacity=64ull*1024*1024;
    std::string name;
    Buffer alloc(VkDeviceSize size,VkBufferUsageFlags usage,VkMemoryPropertyFlags flags) {
        Buffer out;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size=size;bi.usage=usage;bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(device,&bi,nullptr,&out.b));
        VkMemoryRequirements req;vkGetBufferMemoryRequirements(device,out.b,&req);
        uint32_t chosen=UINT32_MAX;
        for(uint32_t i=0;i<memories.memoryTypeCount;i++) {
            if((req.memoryTypeBits&(1u<<i)) && (memories.memoryTypes[i].propertyFlags&flags)==flags) {
                chosen=i;break;
            }
        }
        if(chosen==UINT32_MAX) throw std::runtime_error("No suitable Vulkan memory type");
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize=req.size;ai.memoryTypeIndex=chosen;
        check(vkAllocateMemory(device,&ai,nullptr,&out.mem));
        check(vkBindBufferMemory(device,out.b,out.mem,0));
        if(flags&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) check(vkMapMemory(device,out.mem,0,size,0,&out.map));
        return out;
    }
    void release(Buffer& b) {
        if(b.map)vkUnmapMemory(device,b.mem);
        if(b.b)vkDestroyBuffer(device,b.b,nullptr);
        if(b.mem)vkFreeMemory(device,b.mem,nullptr);
    }
    void begin() {
        check(vkResetCommandBuffer(cmd,0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cmd,&bi));
    }
    void end() {
        check(vkEndCommandBuffer(cmd));check(vkResetFences(device,1,&fence));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};si.commandBufferCount=1;si.pCommandBuffers=&cmd;
        check(vkQueueSubmit(queue,1,&si,fence));
        check(vkWaitForFences(device,1,&fence,VK_TRUE,60000000000ull));
    }
    void barrier(VkPipelineStageFlags src,VkPipelineStageFlags dst,VkAccessFlags from,VkAccessFlags to) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};mb.srcAccessMask=from;mb.dstAccessMask=to;
        vkCmdPipelineBarrier(cmd,src,dst,0,1,&mb,0,nullptr,0,nullptr);
    }
    Engine(const char* shaderPath) {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName="Timestral FP32 linear validation";app.apiVersion=VK_API_VERSION_1_1;
        VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ii.pApplicationInfo=&app;
        check(vkCreateInstance(&ii,nullptr,&instance));
        uint32_t count=0;check(vkEnumeratePhysicalDevices(instance,&count,nullptr));
        std::vector<VkPhysicalDevice> devices(count);check(vkEnumeratePhysicalDevices(instance,&count,devices.data()));
        for(auto p:devices) {
            VkPhysicalDeviceProperties props;vkGetPhysicalDeviceProperties(p,&props);
            if(props.deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {physical=p;name=props.deviceName;break;}
        }
        if(!physical) throw std::runtime_error("No discrete Vulkan device");
        vkGetPhysicalDeviceMemoryProperties(physical,&memories);
        vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,families.data());
        uint32_t family=UINT32_MAX;
        for(uint32_t i=0;i<count;i++)if(families[i].queueFlags&VK_QUEUE_COMPUTE_BIT){family=i;break;}
        if(family==UINT32_MAX)throw std::runtime_error("No compute queue");
        float priority=1.;
        VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qi.queueFamilyIndex=family;qi.queueCount=1;qi.pQueuePriorities=&priority;
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};di.queueCreateInfoCount=1;di.pQueueCreateInfos=&qi;
        check(vkCreateDevice(physical,&di,nullptr,&device));vkGetDeviceQueue(device,family,0,&queue);
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pi.queueFamilyIndex=family;pi.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        check(vkCreateCommandPool(device,&pi,nullptr,&pool));
        VkCommandBufferAllocateInfo ci{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ci.commandPool=pool;ci.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ci.commandBufferCount=1;
        check(vkAllocateCommandBuffers(device,&ci,&cmd));
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};check(vkCreateFence(device,&fi,nullptr,&fence));
        VkDescriptorSetLayoutBinding bindings[3]{};
        for(uint32_t i=0;i<3;i++){bindings[i].binding=i;bindings[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount=1;bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount=3;li.pBindings=bindings;check(vkCreateDescriptorSetLayout(device,&li,nullptr,&dsLayout));
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1024};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets=256;dpi.poolSizeCount=1;dpi.pPoolSizes=&ps;
        check(vkCreateDescriptorPool(device,&dpi,nullptr,&descriptorPool));
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT,0,12};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount=1;pli.pSetLayouts=&dsLayout;pli.pushConstantRangeCount=1;pli.pPushConstantRanges=&push;
        check(vkCreatePipelineLayout(device,&pli,nullptr,&pipelineLayout));
        std::ifstream file(shaderPath,std::ios::binary|std::ios::ate);
        if(!file)throw std::runtime_error("Cannot open shader");
        size_t bytes=file.tellg();file.seekg(0);
        std::vector<uint32_t> code((bytes+3)/4);file.read((char*)code.data(),bytes);
        VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};sm.codeSize=bytes;sm.pCode=code.data();
        check(vkCreateShaderModule(device,&sm,nullptr,&shader));
        VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cp.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cp.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;cp.stage.module=shader;cp.stage.pName="main";cp.layout=pipelineLayout;
        check(vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&cp,nullptr,&pipeline));
        auto local=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        auto host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        input=alloc(capacity,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,local);
        output=alloc(capacity,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,local);
        stageIn=alloc(capacity,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,host);
        stageOut=alloc(capacity,VK_BUFFER_USAGE_TRANSFER_DST_BIT,host);
        fprintf(stderr,"Vulkan FP32 device: %s\n",name.c_str());
    }
    int add(const float* weights,const float* bias,uint32_t n,uint32_t k) {
        VkDeviceSize bytes=(VkDeviceSize(n)*k+n)*4;
        if(bytes>capacity)throw std::runtime_error("Weight exceeds staging capacity");
        Layer layer;layer.n=n;layer.k=k;
        layer.weights=alloc(bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        memcpy(stageIn.map,weights,size_t(n)*k*4);
        if(bias)memcpy((float*)stageIn.map+size_t(n)*k,bias,n*4);
        else memset((float*)stageIn.map+size_t(n)*k,0,n*4);
        begin();VkBufferCopy copy{0,0,bytes};vkCmdCopyBuffer(cmd,stageIn.b,layer.weights.b,1,&copy);
        barrier(VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT);end();
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool=descriptorPool;ai.descriptorSetCount=1;ai.pSetLayouts=&dsLayout;
        check(vkAllocateDescriptorSets(device,&ai,&layer.ds));
        VkDescriptorBufferInfo infos[3]={{input.b,0,capacity},{layer.weights.b,0,bytes},{output.b,0,capacity}};
        VkWriteDescriptorSet writes[3]{};
        for(uint32_t i=0;i<3;i++){
            writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;writes[i].dstSet=layer.ds;writes[i].dstBinding=i;
            writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[i].pBufferInfo=&infos[i];
        }
        vkUpdateDescriptorSets(device,3,writes,0,nullptr);
        layers.push_back(layer);return layers.size()-1;
    }
    void run(int id,const float* x,float* y,uint32_t m) {
        if(id<0 || size_t(id)>=layers.size())throw std::runtime_error("Invalid layer");
        auto& l=layers[id];
        VkDeviceSize inbytes=VkDeviceSize(m)*l.k*4,outbytes=VkDeviceSize(m)*l.n*4;
        if(inbytes>capacity || outbytes>capacity)throw std::runtime_error("Activation exceeds buffer");
        memcpy(stageIn.map,x,inbytes);begin();
        VkBufferCopy incopy{0,0,inbytes};vkCmdCopyBuffer(cmd,stageIn.b,input.b,1,&incopy);
        barrier(VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT);
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipelineLayout,0,1,&l.ds,0,nullptr);
        uint32_t dims[3]={m,l.n,l.k};vkCmdPushConstants(cmd,pipelineLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,12,dims);
        vkCmdDispatch(cmd,(l.n+63)/64,(m+63)/64,1);
        barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferCopy outcopy{0,0,outbytes};vkCmdCopyBuffer(cmd,output.b,stageOut.b,1,&outcopy);
        barrier(VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT);end();
        memcpy(y,stageOut.map,outbytes);
    }
    ~Engine(){
        if(!device)return;
        vkDeviceWaitIdle(device);
        for(auto& l:layers)release(l.weights);
        release(input);release(output);release(stageIn);release(stageOut);
        vkDestroyPipeline(device,pipeline,nullptr);vkDestroyShaderModule(device,shader,nullptr);
        vkDestroyPipelineLayout(device,pipelineLayout,nullptr);vkDestroyDescriptorPool(device,descriptorPool,nullptr);
        vkDestroyDescriptorSetLayout(device,dsLayout,nullptr);vkDestroyFence(device,fence,nullptr);
        vkDestroyCommandPool(device,pool,nullptr);vkDestroyDevice(device,nullptr);vkDestroyInstance(instance,nullptr);
    }
};
extern "C" {
void* vl_create(const char* path) {
    try{return new Engine(path);}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return nullptr;}
}
int vl_add(void* p,const float* w,const float* bias,uint32_t n,uint32_t k) {
    try{return ((Engine*)p)->add(w,bias,n,k);}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return -1;}
}
int vl_run(void* p,int id,const float* x,float* y,uint32_t m) {
    try{((Engine*)p)->run(id,x,y,m);return 0;}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return -1;}
}
void vl_destroy(void* p){delete (Engine*)p;}
}
