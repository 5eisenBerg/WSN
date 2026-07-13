#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/lr-wpan-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/opengym-module.h"
#include <sys/stat.h>
#include <vector>
#include <string>
#include <algorithm>

using namespace ns3;
NS_LOG_COMPONENT_DEFINE ("WSN_DQN_Project_100Nodes");

#define NUM_NODES 100     // Scaled up to a dense 100-node network
#define STATE_SIZE 6     // State size remains optimal: [QueueLen, QueueAge, Energy, Sensed, Threshold, CriticalFlag]
#define ACTION_SIZE 5    // 0: Sink0, 1: Sink1, 2: Sink2, 3: Wait/Buffer, 4: Flush/Drop Packet

struct NodeState {
    double queueLen;
    double queueAge;     // Dynamic aging parameter (USP)
    double energy;       // Track the network node battery life
    double sensedValue;
    double threshold;
};

static NodeState g_states[NUM_NODES];
static uint32_t g_lastNodeIdx = 0;
static NodeContainer g_nodes;
static NetDeviceContainer g_devices;
static Ipv4InterfaceContainer g_interfaces;
static Ptr<OpenGymInterface> g_gym;

void SendPacket (uint32_t srcIdx, uint32_t sinkIdx) {
    Ptr<Packet> pkt = Create<Packet> (100); 
    g_devices.Get(srcIdx)->Send(pkt, g_devices.Get(sinkIdx)->GetAddress(), 0x0800);
}

static Ptr<OpenGymSpace> GetObservationSpace () {
    std::vector<uint32_t> shape = {STATE_SIZE};
    return CreateObject<OpenGymBoxSpace> (0.0, 100.0, shape, TypeNameGet<double>());
}

static Ptr<OpenGymSpace> GetActionSpace () {
    return CreateObject<OpenGymDiscreteSpace> (ACTION_SIZE);
}

static Ptr<OpenGymDataContainer> GetObservation () {
    // Pick a random sensor node (excluding the three scaled sinks at 97, 98, 99)
    g_lastNodeIdx = rand() % (NUM_NODES - 3); 
    uint32_t i = g_lastNodeIdx;
    
    std::vector<uint32_t> shape = {STATE_SIZE};
    Ptr<OpenGymBoxContainer<double>> box = CreateObject<OpenGymBoxContainer<double>> (shape);
    
    // Normalization values passed straight to Python
    box->AddValue (g_states[i].queueLen / 20.0);
    box->AddValue (g_states[i].queueAge / 10.0);  
    box->AddValue (g_states[i].energy / 100.0);
    box->AddValue (g_states[i].sensedValue / 100.0);
    box->AddValue (g_states[i].threshold / 100.0);
    box->AddValue (g_states[i].sensedValue > g_states[i].threshold ? 1.0 : 0.0); 
    
    return box;
}

static float GetReward () { return 0.0; } // Managed on Python side
static bool GetGameOver () { return false; }
static std::string GetExtraInfo () { return ""; }

static bool ExecuteActions (Ptr<OpenGymDataContainer> action) {
    uint32_t i = g_lastNodeIdx;
    uint32_t a = DynamicCast<OpenGymDiscreteContainer>(action)->GetValue();
    
    // Sinks scaled to the edge bounds of the 100-node network topology
    uint32_t sink0 = 97, sink1 = 98, sink2 = 99; 
    
    // 1. Adaptive Data Filtering Bypass (High-Priority Overrule)
    if (g_states[i].sensedValue > g_states[i].threshold) {
        SendPacket(i, sink0); 
        g_states[i].queueLen = std::max(0.0, g_states[i].queueLen - 1.0);
        g_states[i].energy -= 2.0; 
        g_states[i].queueAge = 0.0; 
    } 
    else {
        // 2. Normal Traffic Conditions Managed by DQN
        switch (a) {
            case 0: 
                SendPacket(i, sink0); g_states[i].queueLen = std::max(0.0, g_states[i].queueLen - 2.0); 
                g_states[i].energy -= 1.5; g_states[i].queueAge = 0.0; break;
            case 1: 
                SendPacket(i, sink1); g_states[i].queueLen = std::max(0.0, g_states[i].queueLen - 2.0); 
                g_states[i].energy -= 1.5; g_states[i].queueAge = 0.0; break;
            case 2: 
                SendPacket(i, sink2); g_states[i].queueLen = std::max(0.0, g_states[i].queueLen - 2.0); 
                g_states[i].energy -= 1.5; g_states[i].queueAge = 0.0; break;
            case 3: 
                g_states[i].energy -= 0.01; break; // Intentional buffering wait state
            case 4: 
                g_states[i].queueLen = std::max(0.0, g_states[i].queueLen - 1.0); break; // Memory overflow dropping action
        }
    }

    // Guard safety bounds
    if (g_states[i].queueLen < 0) g_states[i].queueLen = 0;
    if (g_states[i].energy < 0) g_states[i].energy = 0;

    // Environmental state update loop for all 100 nodes
    for (uint32_t n = 0; n < NUM_NODES; n++) {
        g_states[n].queueLen += 0.4; 
        
        // Accumulate age tracking if packets wait inside the buffer
        if (g_states[n].queueLen > 0) {
            g_states[n].queueAge += 0.1;
        } else {
            g_states[n].queueAge = 0.0;
        }
        
        g_states[n].energy -= 0.01;
        if (g_states[n].energy < 0) g_states[n].energy = 0;
        g_states[n].sensedValue = 20 + (rand() % 70);
    }
    return true;
}

void ScheduleNextStep (uint32_t interval) {
    g_gym->NotifyCurrentState ();
    Simulator::Schedule (MilliSeconds (interval), &ScheduleNextStep, interval);
}

int main (int argc, char *argv[]) {
    uint32_t openGymPort = 5555;
    CommandLine cmd;
    cmd.AddValue ("openGymPort", "Port for OpenGym", openGymPort);
    cmd.Parse (argc, argv);
    
    system("mkdir -p scratch/pcaps");
    g_nodes.Create (NUM_NODES);
    
    MobilityHelper mobility;
    mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
    mobility.Install (g_nodes);
    
    LrWpanHelper lrWpanHelper;
    g_devices = lrWpanHelper.Install (g_nodes);
    
    InternetStackHelper stack;
    stack.Install (g_nodes);
    
    Ipv4AddressHelper address;
    address.SetBase ("10.1.1.0", "255.255.255.0");
    g_interfaces = address.Assign (g_devices);
    
    lrWpanHelper.EnablePcapAll (std::string("scratch/pcaps/wsn-100dynamic"), true);
    
    g_gym = CreateObject<OpenGymInterface> (openGymPort);
    g_gym->SetGetObservationSpaceCb (MakeCallback (&GetObservationSpace));
    g_gym->SetGetActionSpaceCb (MakeCallback (&GetActionSpace));
    g_gym->SetGetObservationCb (MakeCallback (&GetObservation));
    g_gym->SetGetRewardCb (MakeCallback (&GetReward));
    g_gym->SetGetGameOverCb (MakeCallback (&GetGameOver));
    g_gym->SetGetExtraInfoCb (MakeCallback (&GetExtraInfo));
    g_gym->SetExecuteActionsCb (MakeCallback (&ExecuteActions));
    
    for(int i = 0; i < NUM_NODES; i++) {
        g_states[i].queueLen = 2; 
        g_states[i].queueAge = 0.0;
        g_states[i].energy = 100; 
        g_states[i].threshold = 60; 
    }
    
    Simulator::Schedule (MilliSeconds (100), &ScheduleNextStep, 100);
    Simulator::Stop (Seconds (60.0));
    Simulator::Run ();
    g_gym->NotifySimulationEnd ();
    Simulator::Destroy ();
    return 0;
}
