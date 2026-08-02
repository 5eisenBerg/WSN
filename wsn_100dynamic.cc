#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/opengym-module.h"
#include "ns3/random-variable-stream.h"
#include <sstream>
#include <deque>
#include <numeric>
#include <cmath>

using namespace ns3;
NS_LOG_COMPONENT_DEFINE("WSN_Final_V50");

#define NUM_NODES 101
#define SENSOR_NODES 100
#define SINK_NODE_ID 100
#define STATE_SIZE 7
#define ACTION_SIZE 5
#define INITIAL_ENERGY 0.5
#define MAX_QUEUE_CAPACITY 30
#define PACKET_BITS 800
#define MAX_SIM_STEPS 5000

const double SENSING_ENERGY_NJ=20.0, IDLE_ENERGY_NJ=10.0, SLEEP_ENERGY_NJ=1.0;
const double HP_TIMEOUT=2.0, NORMAL_TIMEOUT=5.0;
const double ALPHA=0.7, BETA=0.3;
const double E_ELEC=50e-9, E_FS=10e-12, E_MP=0.0013e-12, D0=sqrt(E_FS/E_MP);

struct PacketInfo { Time enqueueTime; };
struct WSNNode { double residualEnergy=INITIAL_ENERGY; std::deque<PacketInfo> normal_q; std::deque<PacketInfo> hp_q; int neighborCount=0; };
static WSNNode g_nodesState[SENSOR_NODES];
static uint32_t g_currentCH=0, g_sim_step_count=0;
static NodeContainer g_allNodes;
static Ptr<OpenGymInterface> g_gym;
static Ptr<UniformRandomVariable> g_rand;
static uint64_t g_packetsGenerated=0, g_packetsDelivered=0, g_packetsDropped=0, g_normalPacketsDelivered=0, g_hpPacketsDelivered=0;
static double g_totalNormalDelay=0.0, g_totalHpDelay=0.0;
static std::string g_per_step_info = "", g_final_info = "";

void ConsumeJouleEnergy(uint32_t n, double nj) {
    if (g_nodesState[n].residualEnergy > 0) {
        g_nodesState[n].residualEnergy = std::max(0.0, g_nodesState[n].residualEnergy - (nj * 1e-9));
    }
}

void ConsumeBitEnergy(uint32_t n, double b, double d) {
    if (g_nodesState[n].residualEnergy > 0) {
        double c = E_ELEC * b;
        if (d > 0) {
            c += (d < D0) ? (E_FS * b * pow(d, 2)) : (E_MP * b * pow(d, 4));
        }
        ConsumeJouleEnergy(n, c * 1e9);
    }
}

void UpdateNetworkTopology() {
    for (uint32_t i = 0; i < SENSOR_NODES; ++i) {
        g_nodesState[i].neighborCount = 0;
        for (uint32_t j = 0; j < SENSOR_NODES; ++j) {
            if (i != j && g_allNodes.Get(i)->GetObject<MobilityModel>()->GetDistanceFrom(g_allNodes.Get(j)->GetObject<MobilityModel>()) < 40.0) {
                g_nodesState[i].neighborCount++;
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

Ptr<OpenGymSpace> GetObservationSpace() {
    return CreateObject<OpenGymBoxSpace>(0.0f, 1.0f, std::vector<uint32_t>{STATE_SIZE}, "float");
}

Ptr<OpenGymSpace> GetActionSpace() {
    return CreateObject<OpenGymDiscreteSpace>(ACTION_SIZE);
}

std::string GetExtraInfo() {
    return g_per_step_info;
}

std::string GetFinalInfo() {
    return g_final_info;
}

bool GetGameOver() {
    return (g_sim_step_count >= MAX_SIM_STEPS);
}

Ptr<OpenGymDataContainer> GetObservation() {
    g_sim_step_count++;
    uint32_t hp_timeout_drops = 0, n_timeout_drops = 0;
    while (!g_nodesState[g_currentCH].hp_q.empty() && (Simulator::Now() - g_nodesState[g_currentCH].hp_q.front().enqueueTime).GetSeconds() > HP_TIMEOUT) {
        g_nodesState[g_currentCH].hp_q.pop_front();
        g_packetsDropped++;
        hp_timeout_drops++;
    }
    while (!g_nodesState[g_currentCH].normal_q.empty() && (Simulator::Now() - g_nodesState[g_currentCH].normal_q.front().enqueueTime).GetSeconds() > NORMAL_TIMEOUT) {
        g_nodesState[g_currentCH].normal_q.pop_front();
        g_packetsDropped++;
        n_timeout_drops++;
    }
    std::stringstream ss;
    ss << "{\"hp_timeout\":" << hp_timeout_drops << ",\"normal_timeout\":" << n_timeout_drops << "}";
    g_per_step_info = ss.str();
    
    int numPackets = g_rand->GetInteger(1, 4);
    for (int i = 0; i < numPackets; ++i) {
        if (g_rand->GetValue() < 0.6) {
            uint32_t taskNode = g_rand->GetInteger(0, SENSOR_NODES - 1);
            int attempts = 0;
            while ((taskNode == g_currentCH || g_nodesState[taskNode].residualEnergy <= 0) && attempts++ < 100) {
                taskNode = g_rand->GetInteger(0, SENSOR_NODES - 1);
            }
            if (attempts < 100) {
                g_packetsGenerated++;
                ConsumeJouleEnergy(taskNode, SENSING_ENERGY_NJ);
                double dist = g_allNodes.Get(taskNode)->GetObject<MobilityModel>()->GetDistanceFrom(g_allNodes.Get(g_currentCH)->GetObject<MobilityModel>());
                ConsumeBitEnergy(taskNode, PACKET_BITS, dist);
                ConsumeBitEnergy(g_currentCH, PACKET_BITS, 0);
                if (g_rand->GetValue() < 0.25) {
                    if (g_nodesState[g_currentCH].hp_q.size() < MAX_QUEUE_CAPACITY) g_nodesState[g_currentCH].hp_q.push_back({Simulator::Now()});
                    else g_packetsDropped++;
                } else {
                    if (g_nodesState[g_currentCH].normal_q.size() < MAX_QUEUE_CAPACITY) g_nodesState[g_currentCH].normal_q.push_back({Simulator::Now()});
                    else g_packetsDropped++;
                }
            }
        }
    }
    
    auto box = CreateObject<OpenGymBoxContainer<double>>(std::vector<uint32_t>{STATE_SIZE});
    double n_age = g_nodesState[g_currentCH].normal_q.empty() ? 0.0 : (Simulator::Now() - g_nodesState[g_currentCH].normal_q.front().enqueueTime).GetSeconds();
    double hp_age = g_nodesState[g_currentCH].hp_q.empty() ? 0.0 : (Simulator::Now() - g_nodesState[g_currentCH].hp_q.front().enqueueTime).GetSeconds();
    double dist_to_sink = g_allNodes.Get(g_currentCH)->GetObject<MobilityModel>()->GetDistanceFrom(g_allNodes.Get(SINK_NODE_ID)->GetObject<MobilityModel>());
    box->AddValue((double)g_nodesState[g_currentCH].normal_q.size() / MAX_QUEUE_CAPACITY);
    box->AddValue((double)g_nodesState[g_currentCH].hp_q.size() / MAX_QUEUE_CAPACITY);
    box->AddValue(g_nodesState[g_currentCH].residualEnergy / INITIAL_ENERGY);
    box->AddValue(std::min(1.0, n_age / NORMAL_TIMEOUT));
    box->AddValue(std::min(1.0, hp_age / HP_TIMEOUT));
    box->AddValue((double)g_nodesState[g_currentCH].neighborCount / (SENSOR_NODES - 1));
    box->AddValue(std::min(1.0, dist_to_sink / 200.0));
    return box;
}

bool ExecuteActions(Ptr<OpenGymDataContainer> a) {
    uint32_t act = DynamicCast<OpenGymDiscreteContainer>(a)->GetValue();
    double dist_to_sink = g_allNodes.Get(g_currentCH)->GetObject<MobilityModel>()->GetDistanceFrom(g_allNodes.Get(SINK_NODE_ID)->GetObject<MobilityModel>());
    uint32_t hp_s = 0, n_s = 0, hp_d = 0, n_d = 0;
    
    if (act == 0 && !g_nodesState[g_currentCH].hp_q.empty()) {
        PacketInfo p = g_nodesState[g_currentCH].hp_q.front();
        g_nodesState[g_currentCH].hp_q.pop_front();
        g_totalHpDelay += (Simulator::Now() - p.enqueueTime).GetSeconds();
        g_hpPacketsDelivered++;
        g_packetsDelivered++;
        ConsumeBitEnergy(g_currentCH, PACKET_BITS, dist_to_sink);
        hp_s = 1;
    } else if (act == 1 && !g_nodesState[g_currentCH].normal_q.empty()) {
        PacketInfo p = g_nodesState[g_currentCH].normal_q.front();
        g_nodesState[g_currentCH].normal_q.pop_front();
        g_totalNormalDelay += (Simulator::Now() - p.enqueueTime).GetSeconds();
        g_normalPacketsDelivered++;
        g_packetsDelivered++;
        ConsumeBitEnergy(g_currentCH, PACKET_BITS, dist_to_sink);
        n_s = 1;
    } else if (act == 2 && !g_nodesState[g_currentCH].normal_q.empty()) {
        g_nodesState[g_currentCH].normal_q.pop_front();
        g_packetsDropped++;
        n_d = 1;
    } else if (act == 3) {
        ConsumeJouleEnergy(g_currentCH, SLEEP_ENERGY_NJ);
    } else if (act == 4 && !g_nodesState[g_currentCH].hp_q.empty()) {
        g_nodesState[g_currentCH].hp_q.pop_front();
        g_packetsDropped++;
        hp_d = 1;
    } else {
        ConsumeJouleEnergy(g_currentCH, IDLE_ENERGY_NJ);
    }
    
    std::string current_info = g_per_step_info;
    current_info.pop_back(); // remove '}'
    std::stringstream ss;
    ss << ",\"hp_sent\":" << hp_s << ",\"normal_sent\":" << n_s << ",\"hp_dropped\":" << hp_d << ",\"normal_dropped\":" << n_d << "}";
    current_info += ss.str();
    g_per_step_info = current_info;
    
    return true;
}

void ScheduleNextStep(uint32_t i, uint32_t& c) {
    if (GetGameOver()) {
        double s = Simulator::Now().GetSeconds();
        double t = (s > 0) ? (g_packetsDelivered * PACKET_BITS / s) / 1000.0 : 0.0;
        double pdr = (g_packetsGenerated > 0) ? ((double)g_packetsDelivered / g_packetsGenerated) * 100.0 : 0.0;
        double plr = (g_packetsGenerated > 0) ? ((double)g_packetsDropped / g_packetsGenerated) * 100.0 : 0.0;
        double nd = (g_normalPacketsDelivered > 0) ? (g_totalNormalDelay / g_normalPacketsDelivered) : 0.0;
        double hd = (g_hpPacketsDelivered > 0) ? (g_totalHpDelay / g_hpPacketsDelivered) : 0.0;
        
        double te = 0;
        for (uint32_t j = 0; j < SENSOR_NODES; ++j) te += g_nodesState[j].residualEnergy;
        double ec = (SENSOR_NODES * INITIAL_ENERGY) - te;
        double eff = (g_packetsDelivered > 0) ? (ec * 1e9) / (g_packetsDelivered * PACKET_BITS) : 0.0;
        
        std::stringstream ss;
        ss << "{\"throughput_kbps\":" << t 
           << ",\"pdr_pct\":" << pdr 
           << ",\"plr_pct\":" << plr 
           << ",\"avg_normal_delay_s\":" << nd 
           << ",\"avg_hp_delay_s\":" << hd 
           << ",\"energy_nj_bit\":" << eff << "}";
        
        g_final_info = ss.str();
        g_gym->NotifySimulationEnd();
        return;
    }
    
    if ((c > 0 && g_nodesState[g_currentCH].residualEnergy < INITIAL_ENERGY * 0.05) || (c > 0 && c % 20 == 0)) {
        ElectClusterHead();
    }
    c++;
    g_gym->NotifyCurrentState();
    Simulator::Schedule(MilliSeconds(100), &ScheduleNextStep, i, c);
}

int main(int argc, char* argv[]) {
    g_sim_step_count = 0;
    uint32_t p = 5555, r = 1;
    CommandLine cmd;
    cmd.AddValue("openGymPort", "Port", p);
    cmd.AddValue("run", "Run", r);
    cmd.Parse(argc, argv);
    
    SeedManager::SetSeed(12345);
    SeedManager::SetRun(r);
    
    g_rand = CreateObject<UniformRandomVariable>();
    g_allNodes.Create(NUM_NODES);
    
    Ptr<ListPositionAllocator> spa = CreateObject<ListPositionAllocator>();
    spa->Add(Vector(50.0, 150.0, 0.0));
    MobilityHelper sm;
    sm.SetPositionAllocator(spa);
    sm.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    sm.Install(g_allNodes.Get(SINK_NODE_ID));
    
    MobilityHelper senm;
    senm.SetPositionAllocator("ns3::RandomRectanglePositionAllocator", "X", StringValue("ns3::UniformRandomVariable[Min=0|Max=100]"), "Y", StringValue("ns3::UniformRandomVariable[Min=0|Max=100]"));
    senm.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    
    NodeContainer sensors;
    for (uint32_t i = 0; i < SENSOR_NODES; ++i) {
        sensors.Add(g_allNodes.Get(i));
    }
    senm.Install(sensors);
    
    for (int i = 0; i < SENSOR_NODES; ++i) {
        g_nodesState[i] = WSNNode();
    }
    
    ElectClusterHead();
    
    g_gym = CreateObject<OpenGymInterface>(p);
    g_gym->SetGetObservationSpaceCb(MakeCallback(&GetObservationSpace));
    g_gym->SetGetActionSpaceCb(MakeCallback(&GetActionSpace));
    g_gym->SetGetObservationCb(MakeCallback(&GetObservation));
    g_gym->SetExecuteActionsCb(MakeCallback(&ExecuteActions));
    g_gym->SetGetGameOverCb(MakeCallback(&GetGameOver));
    g_gym->SetGetExtraInfoCb(MakeCallback(&GetExtraInfo));
    g_gym->AddExtraCallback("GetFinalInfo", MakeCallback(&GetFinalInfo));
    
    uint32_t rc = 0;
    Simulator::Schedule(MilliSeconds(100), &ScheduleNextStep, 100, rc);
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}
