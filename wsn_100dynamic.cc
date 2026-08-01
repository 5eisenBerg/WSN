#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/lr-wpan-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/opengym-module.h"
#include "ns3/random-variable-stream.h"
#include <sstream>
#include <numeric>
#include <cmath> // For std::abs, std::sqrt

using namespace ns3;
NS_LOG_COMPONENT_DEFINE("WSN_Merged_V4_Fixed");

// --- Constants ---
#define NUM_NODES 100
#define SENSOR_NODES (NUM_NODES - 3)
#define INITIAL_ENERGY 0.1
#define MAX_QUEUE_CAPACITY 20
#define PACKET_BITS 800

const double SENSING_ENERGY_NJ = 20.0;
const double ALPHA = 0.7;
const double BETA = 0.3;
const double K1 = 3.0;
const double K2 = 5.0;
const double COMM_RANGE = 40.0;

// --- Heinzelman First-Order Radio Model Parameters ---
const double E_ELEC = 50e-9;
const double E_FS = 10e-12;
const double E_MP = 0.0013e-12;
const double D0 = std::sqrt(E_FS / E_MP);

struct WSNNode {
    double residualEnergy = INITIAL_ENERGY;
    double packetAge = 0.0;
    double currentValue = 50.0;
    double prevValue = 50.0;
    double ewma_mu = 50.0;
    double ewma_sigma_sq = 10.0;
    int normal_q_len = 0;
    int hp_q_len = 0;
    int neighborCount = 0;
};

static WSNNode g_nodesState[NUM_NODES];
static uint32_t g_currentCH = 0;
static NodeContainer g_allNodes;
static Ptr<OpenGymInterface> g_gym;
static Ptr<UniformRandomVariable> g_rand;

// --- Ground-Truth Metric Trackers ---
static uint64_t g_packetsGenerated = 0;
static uint64_t g_packetsDelivered = 0;
static uint64_t g_packetsDropped = 0;
static uint64_t g_normalPacketsDelivered = 0;
static double g_totalDelay = 0.0;

// --- Energy Consumers ---
void ConsumeBitEnergy(uint32_t n, double b, double d) {
    if (g_nodesState[n].residualEnergy > 0.0) {
        double c = E_ELEC * b;
        if (d > 0.0) {
            c += (d < D0) ? (E_FS * b * std::pow(d, 2.0)) : (E_MP * b * std::pow(d, 4.0));
        }
        g_nodesState[n].residualEnergy = std::max(0.0, g_nodesState[n].residualEnergy - c);
    }
}

void ConsumeJouleEnergy(uint32_t n, double nj) {
    if (g_nodesState[n].residualEnergy > 0.0) {
        g_nodesState[n].residualEnergy = std::max(0.0, g_nodesState[n].residualEnergy - (nj * 1e-9));
    }
}

// --- Topology and CH Election Updates ---
void UpdateNetworkTopology() {
    for (uint32_t i = 0; i < SENSOR_NODES; ++i) {
        g_nodesState[i].neighborCount = 0;
        for (uint32_t j = 0; j < SENSOR_NODES; ++j) {
            if (i != j) {
                double dist = g_allNodes.Get(i)->GetObject<MobilityModel>()->GetDistanceFrom(g_allNodes.Get(j)->GetObject<MobilityModel>());
                if (dist < COMM_RANGE) {
                    g_nodesState[i].neighborCount++;
                }
            }
        }
    }
}

void ElectClusterHead() {
    UpdateNetworkTopology();
    double max_w = -1.0;
    uint32_t new_ch = g_currentCH;
    int max_n = 0;
    for (uint32_t i = 0; i < SENSOR_NODES; ++i) {
        if (g_nodesState[i].neighborCount > max_n) {
            max_n = g_nodesState[i].neighborCount;
        }
    }
    for (uint32_t i = 0; i < SENSOR_NODES; ++i) {
        if (g_nodesState[i].residualEnergy > INITIAL_ENERGY * 0.1) {
            double ef = g_nodesState[i].residualEnergy / INITIAL_ENERGY;
            double cf = (max_n > 0) ? (double)g_nodesState[i].neighborCount / max_n : 0.0;
            double w = ALPHA * ef + BETA * cf;
            if (w > max_w) {
                max_w = w;
                new_ch = i;
            }
        }
    }
    if (g_currentCH != new_ch || g_nodesState[g_currentCH].residualEnergy <= 0.01) {
        g_currentCH = new_ch;
    }
}

void UpdateEWMA(uint32_t n, double v, double g = 0.1) {
    g_nodesState[n].prevValue = g_nodesState[n].currentValue;
    g_nodesState[n].currentValue = v;
    g_nodesState[n].ewma_mu = (1.0 - g) * g_nodesState[n].ewma_mu + g * v;
    g_nodesState[n].ewma_sigma_sq = (1.0 - g) * g_nodesState[n].ewma_sigma_sq + g * std::pow(v - g_nodesState[n].ewma_mu, 2.0);
}

// --- FIXED: Explicit Float Literals and Dtype Strings for newer OpenGym API ---
Ptr<OpenGymSpace> GetObservationSpace() { 
    return CreateObject<OpenGymBoxSpace>(0.0f, 1.0f, std::vector<uint32_t>{5}, "float"); 
}

Ptr<OpenGymSpace> GetActionSpace() { 
    return CreateObject<OpenGymDiscreteSpace>(4); 
}

Ptr<OpenGymDataContainer> GetObservation() {
    uint32_t taskNode = g_rand->GetInteger(0, SENSOR_NODES - 1);
    int attempts = 0;
    while ((taskNode == g_currentCH || g_nodesState[taskNode].residualEnergy <= 0.0) && attempts++ < 100) {
        taskNode = g_rand->GetInteger(0, SENSOR_NODES - 1);
    }
    
    g_packetsGenerated++;
    ConsumeJouleEnergy(taskNode, SENSING_ENERGY_NJ);
    
    double value = g_rand->GetValue(20.0, 90.0);
    UpdateEWMA(taskNode, value);
    
    double dist = g_allNodes.Get(taskNode)->GetObject<MobilityModel>()->GetDistanceFrom(g_allNodes.Get(g_currentCH)->GetObject<MobilityModel>());
    ConsumeBitEnergy(taskNode, PACKET_BITS, dist);
    ConsumeBitEnergy(g_currentCH, PACKET_BITS, 0.0);
    
    // Fixed: Explicit double calculations with std namespace
    double sigma = std::sqrt(std::max(1e-9, g_nodesState[taskNode].ewma_sigma_sq));
    if (std::abs(g_nodesState[taskNode].currentValue - g_nodesState[taskNode].prevValue) > K2 * sigma) {
        if (g_nodesState[g_currentCH].hp_q_len < MAX_QUEUE_CAPACITY) {
            g_nodesState[g_currentCH].hp_q_len++;
        } else {
            g_packetsDropped++;
        }
    } else if (std::abs(g_nodesState[taskNode].currentValue - g_nodesState[taskNode].ewma_mu) > K1 * sigma) {
        if (g_nodesState[g_currentCH].normal_q_len < MAX_QUEUE_CAPACITY) {
            g_nodesState[g_currentCH].normal_q_len++;
        } else {
            g_packetsDropped++;
        }
    }
    
    auto box = CreateObject<OpenGymBoxContainer<double>>(std::vector<uint32_t>{5});
    box->AddValue((double)g_nodesState[g_currentCH].normal_q_len / MAX_QUEUE_CAPACITY);
    box->AddValue((double)g_nodesState[g_currentCH].hp_q_len / MAX_QUEUE_CAPACITY);
    box->AddValue(g_nodesState[g_currentCH].residualEnergy / INITIAL_ENERGY);
    box->AddValue(std::min(1.0, g_nodesState[g_currentCH].packetAge / 20.0));
    box->AddValue(g_nodesState[g_currentCH].hp_q_len > 0 ? 1.0 : 0.0);
    return box;
}

bool GetGameOver() {
    double totalEnergy = 0.0;
    for (uint32_t i = 0; i < SENSOR_NODES; ++i) {
        totalEnergy += g_nodesState[i].residualEnergy;
    }
    return totalEnergy <= 0.01;
}

std::string GetExtraInfo() {
    double simTime = Simulator::Now().GetSeconds();
    double tput = (simTime > 0.0) ? (g_packetsDelivered / simTime) : 0.0;
    double pdr = (g_packetsGenerated > 0) ? ((double)g_packetsDelivered / g_packetsGenerated) * 100.0 : 0.0;
    double delay = (g_normalPacketsDelivered > 0) ? (g_totalDelay / g_normalPacketsDelivered) : 0.0;
    
    double totalEnergy = 0.0;
    for (uint32_t i = 0; i < SENSOR_NODES; ++i) {
        totalEnergy += g_nodesState[i].residualEnergy;
    }
    double energy_consumed = (SENSOR_NODES * INITIAL_ENERGY) - totalEnergy;
    double efficiency = (g_packetsDelivered > 0) ? (energy_consumed * 1e9) / (g_packetsDelivered * PACKET_BITS) : 0.0;
    
    std::stringstream ss;
    ss << "{\"throughput\":" << tput 
       << ",\"pdr\":" << pdr 
       << ",\"avg_delay\":" << delay 
       << ",\"energy_efficiency\":" << efficiency << "}";
    return ss.str();
}

bool ExecuteActions(Ptr<OpenGymDataContainer> a) {
    uint32_t act = DynamicCast<OpenGymDiscreteContainer>(a)->GetValue();
    if (g_nodesState[g_currentCH].normal_q_len > 0) {
        g_nodesState[g_currentCH].packetAge += 0.1;
    } else {
        g_nodesState[g_currentCH].packetAge = 0.0;
    }
    
    if (act == 0 && g_nodesState[g_currentCH].hp_q_len > 0) {
        g_nodesState[g_currentCH].hp_q_len--;
        g_packetsDelivered++;
        ConsumeBitEnergy(g_currentCH, PACKET_BITS, 70.0);
    } else if (act == 1 && g_nodesState[g_currentCH].normal_q_len > 0) {
        g_totalDelay += g_nodesState[g_currentCH].packetAge;
        g_normalPacketsDelivered++;
        g_nodesState[g_currentCH].normal_q_len--;
        g_packetsDelivered++;
        g_nodesState[g_currentCH].packetAge = 0.0;
        ConsumeBitEnergy(g_currentCH, PACKET_BITS, 70.0);
    } else if (act == 2 && g_nodesState[g_currentCH].normal_q_len > 0) {
        g_nodesState[g_currentCH].normal_q_len--;
        g_packetsDropped++;
        g_nodesState[g_currentCH].packetAge = 0.0;
    }
    return true;
}

void ScheduleNextStep(uint32_t i, uint32_t& c) {
    if (g_gym->IsGameOver()) {
        g_gym->NotifySimulationEnd();
        return;
    }
    if (g_nodesState[g_currentCH].residualEnergy < INITIAL_ENERGY * 0.05 || (c > 0 && c % 20 == 0)) {
        ElectClusterHead();
    }
    c++;
    g_gym->NotifyCurrentState();
    Simulator::Schedule(MilliSeconds(i), &ScheduleNextStep, i, c);
}

int main(int argc, char* argv[]) {
    CommandLine cmd;
    uint32_t port = 5555;
    cmd.AddValue("openGymPort", "Port", port);
    cmd.Parse(argc, argv);
    
    g_rand = CreateObject<UniformRandomVariable>();
    g_allNodes.Create(NUM_NODES);
    
    MobilityHelper mobility;
    mobility.SetPositionAllocator("ns3::RandomRectanglePositionAllocator", 
                                  "X", StringValue("ns3::UniformRandomVariable[Min=0|Max=100]"), 
                                  "Y", StringValue("ns3::UniformRandomVariable[Min=0|Max=100]"));
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(g_allNodes);
    
    for (int i = 0; i < NUM_NODES; ++i) {
        g_nodesState[i] = WSNNode();
    }
    g_packetsGenerated = 0;
    g_packetsDelivered = 0;
    g_packetsDropped = 0;
    g_totalDelay = 0.0;
    g_normalPacketsDelivered = 0;
    ElectClusterHead();
    
    g_gym = CreateObject<OpenGymInterface>(port);
    g_gym->SetGetObservationSpaceCb(MakeCallback(&GetObservationSpace));
    g_gym->SetGetActionSpaceCb(MakeCallback(&GetActionSpace));
    g_gym->SetGetObservationCb(MakeCallback(&GetObservation));
    g_gym->SetExecuteActionsCb(MakeCallback(&ExecuteActions));
    g_gym->SetGetGameOverCb(MakeCallback(&GetGameOver));
    g_gym->SetGetExtraInfoCb(MakeCallback(&GetExtraInfo));
    
    uint32_t rc = 0;
    Simulator::Schedule(MilliSeconds(100), &ScheduleNextStep, 100, rc);
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}
