#include "ns3/core-module.hh"
#include "ns3/network-module.h"
#include "ns3/lr-wpan-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/opengym-module.h"
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

using namespace ns3;
NS_LOG_COMPONENT_DEFINE ("WSN_DQN_Module1");

// --- Constants based on published models ---
#define NUM_NODES 100
#define NUM_SINKS 3
#define SENSOR_NODES (NUM_NODES - NUM_SINKS)

// --- State and Action for Module 1 ---
// State: [NormalQ_Len, HighPriorityQ_Len, CH_Energy, Packet_Age, Sensed_Delta]
#define STATE_SIZE 5
// Action: 0:Send_HP, 1:Send_Normal, 2:Drop_Normal, 3:Wait
#define ACTION_SIZE 4

// --- First-Order Radio Model Parameters (from Heinzelman et al.) ---
const double E_ELEC = 50e-9;      // 50 nJ/bit
const double E_FS = 10e-12;       // 10 pJ/bit/m^2
const double E_MP = 0.0013e-12;   // 0.0013 pJ/bit/m^4
const double E_DA = 5e-9;         // 5 nJ/bit/signal (Data Aggregation)
const double D0 = sqrt(E_FS / E_MP);
const double PACKET_BITS = 100 * 8; // 100-byte packet

struct QueueState {
    double normalPriorityLen = 0;
    double highPriorityLen = 0;
    double packetAge = 0.0; // Age of the oldest packet in the normal queue
};

struct SensorState {
    double currentValue = 50.0;
    double previousValue = 50.0;
    double dynamicThreshold = 80.0;
    double ewma_mu = 50.0;
    double ewma_sigma_sq = 10.0;
    double ewma_gamma = 0.1; // Smoothing factor
};

struct NodeEnergy {
    double residualEnergy = 1.0; // Start with 1 Joule
};

struct WSNNode {
    QueueState queue;
    SensorState sensor;
    NodeEnergy energy;
    bool isCH = false;
    uint32_t clusterId = 0;
};

static WSNNode g_nodesState[NUM_NODES];
static uint32_t g_currentCH = 0;
static NodeContainer g_allNodes;
static NetDeviceContainer g_allDevices;
static Ptr<OpenGymInterface> g_gym;


// --- Published Formula Implementation: First-Order Radio Model ---
void ConsumeEnergy(uint32_t nodeId, double bits, double distance) {
    double energyConsumed;
    if (distance < D0) {
        energyConsumed = (E_ELEC * bits) + (E_FS * bits * distance * distance);
    } else {
        energyConsumed = (E_ELEC * bits) + (E_MP * bits * distance * distance * distance * distance);
    }
    g_nodesState[nodeId].energy.residualEnergy -= energyConsumed;
    if (g_nodesState[nodeId].energy.residualEnergy < 0) {
        g_nodesState[nodeId].energy.residualEnergy = 0;
    }
}

// --- Published Formula Implementation: EWMA Dynamic Threshold ---
void UpdateDynamicThreshold(uint32_t nodeId) {
    SensorState& s = g_nodesState[nodeId].sensor;
    s.ewma_mu = (1 - s.ewma_gamma) * s.ewma_mu + s.ewma_gamma * s.currentValue;
    s.ewma_sigma_sq = (1 - s.ewma_gamma) * s.ewma_sigma_sq + s.ewma_gamma * pow(s.currentValue - s.ewma_mu, 2);
    // Threshold is 3 standard deviations above the mean
    s.dynamicThreshold = s.ewma_mu + 3 * sqrt(s.ewma_sigma_sq);
}

// Simple distance calculation
double GetDistance(uint32_t nodeA, uint32_t nodeB) {
    Ptr<MobilityModel> mobA = g_allNodes.Get(nodeA)->GetObject<MobilityModel>();
    Ptr<MobilityModel> mobB = g_allNodes.Get(nodeB)->GetObject<MobilityModel>();
    return mobA->GetDistanceFrom(mobB);
}

// --- Custom Formula (Inspired by LEACH/HEED): Elect a new CH ---
void ElectClusterHead() {
    double maxWeight = -1.0;
    uint32_t newCH = 0;

    for (uint32_t i = 0; i < SENSOR_NODES; ++i) {
        if (g_nodesState[i].energy.residualEnergy > 0) {
            double energyFactor = g_nodesState[i].energy.residualEnergy / 1.0; // Normalized energy
            // For simplicity, we omit the V_i/N factor for now to focus on the DQN at the CH
            double weight = energyFactor; // Simplified for Module 1 focus
            if (weight > maxWeight) {
                maxWeight = weight;
                newCH = i;
            }
        }
    }
    
    for (uint32_t i=0; i < SENSOR_NODES; ++i) g_nodesState[i].isCH = false;
    g_nodesState[newCH].isCH = true;
    g_currentCH = newCH;
    NS_LOG_INFO("New Cluster Head Elected: Node " << g_currentCH);
}

// --- Gym Interface Functions ---
Ptr<OpenGymSpace> GetObservationSpace() {
    std::vector<uint32_t> shape = {STATE_SIZE};
    return CreateObject<OpenGymBoxSpace>(0.0, 1.0, shape, TypeNameGet<double>()); // Normalized space
}

Ptr<OpenGymSpace> GetActionSpace() {
    return CreateObject<OpenGymDiscreteSpace>(ACTION_SIZE);
}

Ptr<OpenGymDataContainer> GetObservation() {
    uint32_t i = g_currentCH;
    std::vector<uint32_t> shape = {STATE_SIZE};
    Ptr<OpenGymBoxContainer<double>> box = CreateObject<OpenGymBoxContainer<double>>(shape);
    
    // State: [NormalQ_Len, HighPriorityQ_Len, CH_Energy, Packet_Age, Sensed_Delta]
    box->AddValue(g_nodesState[i].queue.normalPriorityLen / 20.0); // Normalize by max queue size
    box->AddValue(g_nodesState[i].queue.highPriorityLen / 20.0);
    box->AddValue(g_nodesState[i].energy.residualEnergy / 1.0);
    box->AddValue(g_nodesState[i].queue.packetAge / 10.0); // Normalize by max age
    
    double sensed_delta = abs(g_nodesState[i].sensor.currentValue - g_nodesState[i].sensor.previousValue);
    box->AddValue(sensed_delta / 50.0); // Normalize by expected max delta

    return box;
}

float GetReward() { return 0.0; } // Reward is calculated entirely on the Python side
bool GetGameOver() { return g_nodesState[g_currentCH].energy.residualEnergy <= 0; }
std::string GetExtraInfo() { return ""; }

bool ExecuteActions(Ptr<OpenGymDataContainer> action) {
    uint32_t ch_id = g_currentCH;
    uint32_t a = DynamicCast<OpenGymDiscreteContainer>(action)->GetValue();

    QueueState& q = g_nodesState[ch_id].queue;

    switch (a) {
        case 0: // Send High Priority Packet
            if (q.highPriorityLen > 0) {
                q.highPriorityLen--;
                ConsumeEnergy(ch_id, PACKET_BITS, 50.0); // Assume 50m to sink
            }
            break;
        case 1: // Send Normal Priority Packet
            if (q.normalPriorityLen > 0) {
                q.normalPriorityLen--;
                ConsumeEnergy(ch_id, PACKET_BITS, 50.0);
                q.packetAge = 0; // Reset age
            }
            break;
        case 2: // Drop Normal Priority Packet
            if (q.normalPriorityLen > 0) {
                q.normalPriorityLen--;
                q.packetAge = 0;
            }
            break;
        case 3: // Wait
            // Conserve energy, do nothing.
            break;
    }
    
    // --- Environment Dynamics Update ---
    // 1. All sensor nodes sense data and send to CH
    for (uint32_t i = 0; i < SENSOR_NODES; ++i) {
        if (i == ch_id || g_nodesState[i].energy.residualEnergy <= 0) continue;
        
        g_nodesState[i].sensor.previousValue = g_nodesState[i].sensor.currentValue;
        g_nodesState[i].sensor.currentValue = 20 + (rand() % 70); // Simulate sensing
        ConsumeEnergy(i, 20, 0); // Energy for sensing
        
        // Send data to CH
        double dist = GetDistance(i, ch_id);
        ConsumeEnergy(i, PACKET_BITS, dist);
        
        // CH receives data
        ConsumeEnergy(ch_id, PACKET_BITS, 0); // Energy for reception
        
        // --- CH Data Filtering Logic ---
        UpdateDynamicThreshold(i);
        // High Priority: sudden spike in data
        if (abs(g_nodesState[i].sensor.currentValue - g_nodesState[i].sensor.previousValue) > 15) {
             g_nodesState[ch_id].queue.highPriorityLen++;
        }
        // Normal Priority: data is above the dynamic baseline
        else if (g_nodesState[i].sensor.currentValue > g_nodesState[i].sensor.dynamicThreshold) {
            g_nodesState[ch_id].queue.normalPriorityLen++;
        }
        // Else: data is redundant and is dropped (no queueing)
    }

    // 2. Update packet age in normal queue
    if (g_nodesState[ch_id].queue.normalPriorityLen > 0) {
        g_nodesState[ch_id].queue.packetAge += 0.1;
    } else {
        g_nodesState[ch_id].queue.packetAge = 0.0;
    }
    
    return true;
}

void ScheduleNextStep(uint32_t interval, uint32_t& roundCounter) {
    if (roundCounter % 20 == 0) { // Re-elect CH every 20 rounds
        ElectClusterHead();
    }
    roundCounter++;
    g_gym->NotifyCurrentState();
    Simulator::Schedule(MilliSeconds(interval), &ScheduleNextStep, interval, roundCounter);
}

int main(int argc, char* argv[]) {
    uint32_t openGymPort = 5555;
    CommandLine cmd;
    cmd.AddValue("openGymPort", "Port for OpenGym", openGymPort);
    cmd.Parse(argc, argv);
    
    g_allNodes.Create(NUM_NODES);
    
    MobilityHelper mobility;
    ObjectFactory pos;
    pos.SetTypeId("ns3::RandomRectanglePositionAllocator");
    pos.Set("X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=100.0]"));
    pos.Set("Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=100.0]"));
    Ptr<PositionAllocator> taPositionAlloc = pos.Create()->GetObject<PositionAllocator>();
    mobility.SetPositionAllocator(taPositionAlloc);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(g_allNodes);
    
    LrWpanHelper lrWpanHelper;
    g_allDevices = lrWpanHelper.Install(g_allNodes);
    
    g_gym = CreateObject<OpenGymInterface>(openGymPort);
    g_gym->SetGetObservationSpaceCb(MakeCallback(&GetObservationSpace));
    g_gym->SetGetActionSpaceCb(MakeCallback(&GetActionSpace));
    g_gym->SetGetObservationCb(MakeCallback(&GetObservation));
    g_gym->SetGetRewardCb(MakeCallback(&GetReward));
    g_gym->SetGetGameOverCb(MakeCallback(&GetGameOver));
    g_gym->SetGetExtraInfoCb(MakeCallback(&GetExtraInfo));
    g_gym->SetExecuteActionsCb(MakeCallback(&ExecuteActions));
    
    ElectClusterHead(); // Initial election
    
    uint32_t roundCounter = 0;
    Simulator::Schedule(MilliSeconds(100), &ScheduleNextStep, 100, roundCounter);
    Simulator::Stop(Seconds(60.0));
    Simulator::Run();
    g_gym->NotifySimulationEnd();
    Simulator::Destroy();
    return 0;
}
