// #include "ns3/core-module.h"
// #include "ns3/network-module.h"
// #include "ns3/lr-wpan-module.h"
// #include "ns3/mobility-module.h"
// #include "ns3/internet-module.h"
// #include "ns3/ipv4-address-helper.h"
// #include "ns3/opengym-module.h"
// #include <vector>
// #include <string>
// #include <algorithm>
// #include <cmath>

// using namespace ns3;
// NS_LOG_COMPONENT_DEFINE("WSN_DQN_Module1_Fixed");

// // --- Constants based on published models ---
// #define NUM_NODES 100
// #define NUM_SINKS 3
// #define SENSOR_NODES (NUM_NODES - NUM_SINKS)
// #define STATE_SIZE 5
// #define ACTION_SIZE 4

// // --- First-Order Radio Model Parameters (from Heinzelman et al.) ---
// const double E_ELEC = 50e-9;
// const double E_FS = 10e-12;
// const double E_MP = 0.0013e-12;
// const double D0 = sqrt(E_FS / E_MP);
// const double PACKET_BITS = 100 * 8;

// // --- Data Structures for Node State ---
// struct QueueState {
//     double normalPriorityLen = 0;
//     double highPriorityLen = 0;
//     double packetAge = 0.0;
// };
// struct SensorState {
//     double currentValue = 50.0;
//     double dynamicThreshold = 80.0;
//     double ewma_mu = 50.0;
//     double ewma_sigma_sq = 10.0;
// };
// struct NodeEnergy {
//     double residualEnergy = 1.0;
// };
// struct WSNNode {
//     QueueState queue;
//     SensorState sensor;
//     NodeEnergy energy;
//     bool isCH = false;
// };

// static WSNNode g_nodesState[NUM_NODES];
// static uint32_t g_currentCH = 0;
// static uint32_t g_taskNode = 0; // NEW: The node whose data is being processed
// static NodeContainer g_allNodes;
// static Ptr<OpenGymInterface> g_gym;

// // --- Helper Functions ---
// void ConsumeEnergy(uint32_t nodeId, double bits, double distance) {
//     if (g_nodesState[nodeId].energy.residualEnergy <= 0) return;
//     double energyConsumed = (E_ELEC * bits);
//     if (distance > 0) {
//         energyConsumed += (distance < D0) ? (E_FS * bits * pow(distance, 2))
//                                           : (E_MP * bits * pow(distance, 4));
//     }
//     g_nodesState[nodeId].energy.residualEnergy -= energyConsumed;
// }

// void UpdateDynamicThreshold(uint32_t nodeId, double gamma = 0.1) {
//     SensorState& s = g_nodesState[nodeId].sensor;
//     s.ewma_mu = (1 - gamma) * s.ewma_mu + gamma * s.currentValue;
//     s.ewma_sigma_sq = (1 - gamma) * s.ewma_sigma_sq + gamma * pow(s.currentValue - s.ewma_mu, 2);
//     s.dynamicThreshold = s.ewma_mu + 3 * sqrt(s.ewma_sigma_sq);
// }

// double GetDistance(uint32_t nodeA, uint32_t nodeB) {
//     Ptr<MobilityModel> mobA = g_allNodes.Get(nodeA)->GetObject<MobilityModel>();
//     Ptr<MobilityModel> mobB = g_allNodes.Get(nodeB)->GetObject<MobilityModel>();
//     return mobA->GetDistanceFrom(mobB);
// }

// void ElectClusterHead() {
//     double maxEnergy = -1.0;
//     uint32_t newCH = 0;
//     for (uint32_t i = 0; i < SENSOR_NODES; ++i) {
//         if (g_nodesState[i].energy.residualEnergy > maxEnergy) {
//             maxEnergy = g_nodesState[i].energy.residualEnergy;
//             newCH = i;
//         }
//     }
//     if (g_currentCH != newCH) {
//          for (uint32_t i=0; i < SENSOR_NODES; ++i) g_nodesState[i].isCH = false;
//          g_nodesState[newCH].isCH = true;
//          g_currentCH = newCH;
//     }
// }

// // --- Gym Interface Functions ---
// Ptr<OpenGymSpace> GetObservationSpace() {
//     return CreateObject<OpenGymBoxSpace>(0.0, 1.0, std::vector<uint32_t>{STATE_SIZE}, TypeNameGet<double>());
// }

// Ptr<OpenGymSpace> GetActionSpace() {
//     return CreateObject<OpenGymDiscreteSpace>(ACTION_SIZE);
// }

// // *** MAJOR CHANGE HERE ***
// // The environment now simulates one node sensing data and sending it to the CH.
// // The observation is the state of the CH *after* receiving this data.
// Ptr<OpenGymDataContainer> GetObservation() {
//     // 1. A random sensor node generates data
//     g_taskNode = rand() % SENSOR_NODES;
//     while(g_taskNode == g_currentCH || g_nodesState[g_taskNode].energy.residualEnergy <=0) {
//         g_taskNode = rand() % SENSOR_NODES;
//     }
    
//     // 2. Node senses and sends to CH
//     double prevValue = g_nodesState[g_taskNode].sensor.currentValue;
//     g_nodesState[g_taskNode].sensor.currentValue = 20 + (rand() % 70);
//     ConsumeEnergy(g_taskNode, 20, 0); // Sensing energy
//     ConsumeEnergy(g_taskNode, PACKET_BITS, GetDistance(g_taskNode, g_currentCH)); // Tx energy
//     ConsumeEnergy(g_currentCH, PACKET_BITS, 0); // CH Rx energy
    
//     // 3. CH classifies the incoming packet and queues it
//     UpdateDynamicThreshold(g_taskNode);
//     if (abs(g_nodesState[g_taskNode].sensor.currentValue - prevValue) > 20) { // Spike
//          g_nodesState[g_currentCH].queue.highPriorityLen++;
//     } else if (g_nodesState[g_taskNode].sensor.currentValue > g_nodesState[g_taskNode].sensor.dynamicThreshold) { // Normal
//         g_nodesState[g_currentCH].queue.normalPriorityLen++;
//     } // Otherwise, data is dropped (redundant)

//     // 4. Return the CH's current state to the Python agent for a decision
//     Ptr<OpenGymBoxContainer<double>> box = CreateObject<OpenGymBoxContainer<double>>(std::vector<uint32_t>{STATE_SIZE});
//     box->AddValue(std::min(1.0, g_nodesState[g_currentCH].queue.normalPriorityLen / 20.0));
//     box->AddValue(std::min(1.0, g_nodesState[g_currentCH].queue.highPriorityLen / 20.0));
//     box->AddValue(std::max(0.0, g_nodesState[g_currentCH].energy.residualEnergy / 1.0));
//     box->AddValue(std::min(1.0, g_nodesState[g_currentCH].queue.packetAge / 10.0));
//     box->AddValue(g_nodesState[g_currentCH].queue.highPriorityLen > 0 ? 1.0 : 0.0); // Is there a critical packet?
    
//     return box;
// }

// float GetReward() { return 0.0; }
// bool GetGameOver() { return g_nodesState[g_currentCH].energy.residualEnergy <= 0.05; } // End if CH energy is critical
// std::string GetExtraInfo() { return ""; }

// bool ExecuteActions(Ptr<OpenGymDataContainer> action) {
//     uint32_t ch_id = g_currentCH;
//     uint32_t a = DynamicCast<OpenGymDiscreteContainer>(action)->GetValue();
//     QueueState& q = g_nodesState[ch_id].queue;

//     switch (a) {
//         case 0: // Send High Priority
//             if (q.highPriorityLen > 0) { q.highPriorityLen--; ConsumeEnergy(ch_id, PACKET_BITS, 70.0); } break;
//         case 1: // Send Normal
//             if (q.normalPriorityLen > 0) { q.normalPriorityLen--; q.packetAge = 0; ConsumeEnergy(ch_id, PACKET_BITS, 70.0); } break;
//         case 2: // Drop Normal
//             if (q.normalPriorityLen > 0) { q.normalPriorityLen--; q.packetAge = 0; } break;
//         case 3: // Wait
//              ConsumeEnergy(ch_id, 1, 0); break; // Small energy cost for being idle
//     }

//     // Update packet age
//     if (g_nodesState[ch_id].queue.normalPriorityLen > 0) g_nodesState[ch_id].queue.packetAge += 0.2;
//     else g_nodesState[ch_id].queue.packetAge = 0.0;
    
//     return true;
// }

// void ScheduleNextStep(uint32_t interval, uint32_t& roundCounter) {
//     if (g_gym->IsGameOver()) return;
//     if (roundCounter > 0 && roundCounter % 20 == 0) { ElectClusterHead(); }
//     roundCounter++;
//     g_gym->NotifyCurrentState();
//     Simulator::Schedule(MilliSeconds(interval), &ScheduleNextStep, interval, roundCounter);
// }

// // Main function remains largely the same...
// int main(int argc, char* argv[]) {
//     uint32_t openGymPort = 5555;
//     CommandLine cmd;
//     cmd.AddValue("openGymPort", "Port for OpenGym", openGymPort);
//     cmd.Parse(argc, argv);
    
//     g_allNodes.Create(NUM_NODES);
    
//     MobilityHelper mobility;
//     ObjectFactory pos;
//     pos.SetTypeId("ns3::RandomRectanglePositionAllocator");
//     pos.Set("X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=100.0]"));
//     pos.Set("Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=100.0]"));
//     mobility.SetPositionAllocator(pos.Create()->GetObject<PositionAllocator>());
//     mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
//     mobility.Install(g_allNodes);
    
//     for(int i = 0; i < NUM_NODES; ++i) g_nodesState[i] = WSNNode(); // Reset states

//     ElectClusterHead();
    
//     g_gym = CreateObject<OpenGymInterface>(openGymPort);
//     g_gym->SetGetObservationSpaceCb(MakeCallback(&GetObservationSpace));
//     g_gym->SetGetActionSpaceCb(MakeCallback(&GetActionSpace));
//     g_gym->SetGetObservationCb(MakeCallback(&GetObservation));
//     g_gym->SetGetRewardCb(MakeCallback(&GetReward));
//     g_gym->SetGetGameOverCb(MakeCallback(&GetGameOver));
//     g_gym->SetExecuteActionsCb(MakeCallback(&ExecuteActions));
    
//     uint32_t roundCounter = 0;
//     Simulator::Schedule(MilliSeconds(100), &ScheduleNextStep, 100, roundCounter);
//     Simulator::Stop(Seconds(25.0)); // Episodes are shorter now
//     Simulator::Run();
//     g_gym->NotifySimulationEnd();
//     Simulator::Destroy();
//     return 0;
// }


#include "ns3/core-module.h"
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
NS_LOG_COMPONENT_DEFINE("WSN_DQN_Module1_Advanced");

// --- Constants ---
#define NUM_NODES 100
#define NUM_SINKS 3
#define SENSOR_NODES (NUM_NODES - NUM_SINKS)
#define STATE_SIZE 5
#define ACTION_SIZE 4
#define INITIAL_ENERGY 1.0 // Joules

// --- Academic Formula Constants ---
const double E_ELEC = 50e-9;
const double E_FS = 10e-12;
const double E_MP = 0.0013e-12;
const double D0 = sqrt(E_FS / E_MP);
const double PACKET_BITS = 100 * 8;
const double ALPHA = 0.7; // Weight for energy in CH selection
const double BETA = 0.3;  // Weight for centrality in CH selection
const double K1 = 3.0;    // Multiplier for normal data threshold
const double K2 = 5.0;    // Multiplier for spike detection threshold
const double COMM_RANGE = 40.0; // Communication range in meters for neighbor calculation

// --- Data Structures for Node State ---
struct WSNNode {
    // Queues
    double normalPriorityLen = 0;
    double highPriorityLen = 0;
    double packetAge = 0.0;
    // Sensor
    double currentValue = 50.0;
    double prevValue = 50.0;
    double ewma_mu = 50.0;
    double ewma_sigma_sq = 10.0;
    // Energy
    double residualEnergy = INITIAL_ENERGY;
    // Topology
    bool isCH = false;
    int neighborCount = 0;
};

static WSNNode g_nodesState[NUM_NODES];
static uint32_t g_currentCH = 0;
static NodeContainer g_allNodes;
static Ptr<OpenGymInterface> g_gym;

// --- Helper Functions ---
void ConsumeEnergy(uint32_t nodeId, double bits, double distance) {
    if (g_nodesState[nodeId].residualEnergy <= 0) return;
    double energyConsumed = E_ELEC * bits;
    if (distance > 0) {
        energyConsumed += (distance < D0) ? (E_FS * bits * pow(distance, 2)) : (E_MP * bits * pow(distance, 4));
    }
    g_nodesState[nodeId].residualEnergy -= energyConsumed;
}

double GetDistance(uint32_t nodeA, uint32_t nodeB) {
    Ptr<MobilityModel> mobA = g_allNodes.Get(nodeA)->GetObject<MobilityModel>();
    Ptr<MobilityModel> mobB = g_allNodes.Get(nodeB)->GetObject<MobilityModel>();
    return mobA->GetDistanceFrom(mobB);
}

// *** NEW FUNCTION: Calculates neighbors for all nodes ***
void UpdateNetworkTopology() {
    for (uint32_t i = 0; i < SENSOR_NODES; ++i) {
        g_nodesState[i].neighborCount = 0;
        for (uint32_t j = 0; j < SENSOR_NODES; ++j) {
            if (i == j) continue;
            if (GetDistance(i, j) < COMM_RANGE) {
                g_nodesState[i].neighborCount++;
            }
        }
    }
}

// *** UPGRADED FUNCTION: Implements the PDF's formula ***
void ElectClusterHead() {
    UpdateNetworkTopology(); // Update neighbor counts before election
    
    double maxWeight = -1.0;
    uint32_t newCH = g_currentCH;

    for (uint32_t i = 0; i < SENSOR_NODES; ++i) {
        if (g_nodesState[i].residualEnergy > 0) {
            double energyFactor = g_nodesState[i].residualEnergy / INITIAL_ENERGY;
            double centralityFactor = (SENSOR_NODES > 1) ? (double)g_nodesState[i].neighborCount / (SENSOR_NODES - 1) : 0.0;
            
            // PDF Formula: W = α·(E_res/E_max) + β·(V_i/N)
            double weight = ALPHA * energyFactor + BETA * centralityFactor;

            if (weight > maxWeight) {
                maxWeight = weight;
                newCH = i;
            }
        }
    }
    
    if (g_currentCH != newCH) {
         for (uint32_t i=0; i < SENSOR_NODES; ++i) g_nodesState[i].isCH = false;
         g_nodesState[newCH].isCH = true;
         g_currentCH = newCH;
    }
}

// --- Gym Interface Functions ---
Ptr<OpenGymSpace> GetObservationSpace() {
    return CreateObject<OpenGymBoxSpace>(0.0, 1.0, std::vector<uint32_t>{STATE_SIZE}, TypeNameGet<double>());
}
Ptr<OpenGymSpace> GetActionSpace() {
    return CreateObject<OpenGymDiscreteSpace>(ACTION_SIZE);
}

Ptr<OpenGymDataContainer> GetObservation() {
    // 1. A random sensor node generates data
    uint32_t taskNode = rand() % SENSOR_NODES;
    while(taskNode == g_currentCH || g_nodesState[taskNode].residualEnergy <= 0) {
        taskNode = rand() % SENSOR_NODES;
    }
    
    // 2. Node senses and sends to CH
    g_nodesState[taskNode].prevValue = g_nodesState[taskNode].currentValue;
    g_nodesState[taskNode].currentValue = 20 + (rand() % 70);
    ConsumeEnergy(taskNode, 20, 0); // Sensing energy
    ConsumeEnergy(taskNode, PACKET_BITS, GetDistance(taskNode, g_currentCH)); // Tx energy
    ConsumeEnergy(g_currentCH, PACKET_BITS, 0); // CH Rx energy
    
    // 3. CH classifies the incoming packet
    double spike_delta = abs(g_nodesState[taskNode].currentValue - g_nodesState[taskNode].prevValue);
    double sigma = sqrt(g_nodesState[taskNode].ewma_sigma_sq);
    
    // *** UPGRADED RULE: Using dynamic thresholds from PDF ***
    // High Priority: Spike detected based on dynamic variance
    if (spike_delta > K2 * sigma) {
         g_nodesState[g_currentCH].highPriorityLen++;
    } 
    // Normal Priority: Data is above the dynamic baseline
    else if (abs(g_nodesState[taskNode].currentValue - g_nodesState[taskNode].ewma_mu) > K1 * sigma) {
        g_nodesState[g_currentCH].normalPriorityLen++;
    }

    // 4. Return the CH's current state to the Python agent
    Ptr<OpenGymBoxContainer<double>> box = CreateObject<OpenGymBoxContainer<double>>(std::vector<uint32_t>{STATE_SIZE});
    box->AddValue(std::min(1.0, g_nodesState[g_currentCH].normalPriorityLen / 20.0));
    box->AddValue(std::min(1.0, g_nodesState[g_currentCH].highPriorityLen / 20.0));
    box->AddValue(std::max(0.0, g_nodesState[g_currentCH].residualEnergy / INITIAL_ENERGY));
    box->AddValue(std::min(1.0, g_nodesState[g_currentCH].packetAge / 10.0));
    box->AddValue(g_nodesState[g_currentCH].highPriorityLen > 0 ? 1.0 : 0.0);
    
    return box;
}

float GetReward() { return 0.0; }
bool GetGameOver() { return g_nodesState[g_currentCH].residualEnergy <= (INITIAL_ENERGY * 0.05); }
std::string GetExtraInfo() { return ""; }

bool ExecuteActions(Ptr<OpenGymDataContainer> action) {
    uint32_t ch_id = g_currentCH;
    uint32_t a = DynamicCast<OpenGymDiscreteContainer>(action)->GetValue();
    
    switch (a) {
        case 0: // Send High Priority
            if (g_nodesState[ch_id].highPriorityLen > 0) { g_nodesState[ch_id].highPriorityLen--; ConsumeEnergy(ch_id, PACKET_BITS, 70.0); } break;
        case 1: // Send Normal
            if (g_nodesState[ch_id].normalPriorityLen > 0) { g_nodesState[ch_id].normalPriorityLen--; g_nodesState[ch_id].packetAge = 0; ConsumeEnergy(ch_id, PACKET_BITS, 70.0); } break;
        case 2: // Drop Normal
            if (g_nodesState[ch_id].normalPriorityLen > 0) { g_nodesState[ch_id].normalPriorityLen--; g_nodesState[ch_id].packetAge = 0; } break;
        case 3: // Wait
             ConsumeEnergy(ch_id, 1, 0); break;
    }

    if (g_nodesState[ch_id].normalPriorityLen > 0) g_nodesState[ch_id].packetAge += 0.2;
    else g_nodesState[ch_id].packetAge = 0.0;
    
    return true;
}

void ScheduleNextStep(uint32_t interval, uint32_t& roundCounter) {
    if (g_gym->IsGameOver()) return;
    if (roundCounter > 0 && roundCounter % 20 == 0) { ElectClusterHead(); }
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
    mobility.SetPositionAllocator("ns3::RandomRectanglePositionAllocator",
                                  "X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=100.0]"),
                                  "Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=100.0]"));
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(g_allNodes);
    
    for(int i = 0; i < NUM_NODES; ++i) g_nodesState[i] = WSNNode();

    ElectClusterHead();
    
    g_gym = CreateObject<OpenGymInterface>(openGymPort);
    g_gym->SetGetObservationSpaceCb(MakeCallback(&GetObservationSpace));
    g_gym->SetGetActionSpaceCb(MakeCallback(&GetActionSpace));
    g_gym->SetGetObservationCb(MakeCallback(&GetObservation));
    g_gym->SetGetRewardCb(MakeCallback(&GetReward));
    g_gym->SetGetGameOverCb(MakeCallback(&GetGameOver));
    g_gym->SetExecuteActionsCb(MakeCallback(&ExecuteActions));
    
    uint32_t roundCounter = 0;
    Simulator::Schedule(MilliSeconds(100), &ScheduleNextStep, 100, roundCounter);
    Simulator::Stop(Seconds(25.0));
    Simulator::Run();
    g_gym->NotifySimulationEnd();
    Simulator::Destroy();
    return 0;
}
