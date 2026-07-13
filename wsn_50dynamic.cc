#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/lr-wpan-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/opengym-module.h"

#include "wsn_framework/constants.h"
#include "wsn_framework/node_state.h"
#include "wsn_framework/energy_model.h"

#include <sys/stat.h>
#include <vector>
#include <string>
#include <algorithm>

using namespace ns3;
using namespace WSNFramework;

NS_LOG_COMPONENT_DEFINE ("WSN_DQN_Project");

constexpr uint32_t NUM_NODES = WSNConfig::NUM_NODES;
constexpr uint32_t STATE_SIZE = WSNConfig::STATE_SIZE;
constexpr uint32_t ACTION_SIZE = WSNConfig::ACTION_SIZE;

constexpr uint32_t PACKET_SIZE_BITS = WSNConfig::PACKET_SIZE * WSNConfig::BITS_PER_BYTE;

static NodeState g_states[NUM_NODES];
static uint32_t g_lastNodeIdx = 0;
static NodeContainer g_nodes;
static NetDeviceContainer g_devices;
static Ipv4InterfaceContainer g_interfaces;
static Ptr<OpenGymInterface> g_gym;

void SendPacket (uint32_t srcIdx, uint32_t sinkIdx) {
    Ptr<Packet> pkt = Create<Packet>(WSNConfig::PACKET_SIZE);
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
    // Pick a random sensor node (excluding the three sinks at 47, 48, 49)
    g_lastNodeIdx = rand() % (NUM_NODES - 3); 
    uint32_t i = g_lastNodeIdx;
    
    std::vector<uint32_t> shape = {STATE_SIZE};
    Ptr<OpenGymBoxContainer<double>> box = CreateObject<OpenGymBoxContainer<double>> (shape);
    
    // Normalization values passed straight to Python
    box->AddValue(
        static_cast<double>(g_states[i].queue.queueLength) /
        WSNConfig::MAX_QUEUE_SIZE);

    box->AddValue(
        g_states[i].queue.queueWaitingTime /
        WSNConfig::PACKET_TIMEOUT);

    box->AddValue(
        EnergyManager::GetRemainingEnergyPercentage(g_states[i]) /
        100.0);

    box->AddValue(
        g_states[i].sensor.currentValue /
        WSNConfig::SENSOR_MAX);

    box->AddValue(
        g_states[i].sensor.dynamicThreshold /
        WSNConfig::SENSOR_MAX);

    box->AddValue(
        g_states[i].sensor.criticalEvent ? 1.0 : 0.0);
    
    return box;
}

static float GetReward () {
    // Handled on the Python side inside train_dqn.py
    return 0.0;
}

static bool GetGameOver () { return false; }
static std::string GetExtraInfo () { return ""; }

static bool ExecuteActions (Ptr<OpenGymDataContainer> action) {
    uint32_t i = g_lastNodeIdx;
    uint32_t a = DynamicCast<OpenGymDiscreteContainer>(action)->GetValue();
    constexpr uint32_t sink0 = NUM_NODES-3;
    constexpr uint32_t sink1 = NUM_NODES-2;
    constexpr uint32_t sink2 = NUM_NODES-1;
    
    // 1. Adaptive Data Filtering Bypass (High-Priority Overrule)
    if (g_states[i].sensor.currentValue > g_states[i].sensor.dynamicThreshold){
        SendPacket(i, sink0); 
        if (g_states[i].queue.queueLength > 0)
        {
            g_states[i].queue.queueLength--;
        }
        EnergyManager::ConsumeTransmissionEnergy(g_states[i], PACKET_SIZE_BITS, WSNConfig::DEFAULT_TX_DISTANCE);
        g_states[i].queue.queueWaitingTime = 0.0; 
    } 
    else {
        // 2. Normal Traffic Conditions Managed by DQN
        switch (a) {
            case 0:
                SendPacket(i, sink0);

                if (g_states[i].queue.queueLength >= 2)
                {
                    g_states[i].queue.queueLength -= 2;
                }
                else
                {
                    g_states[i].queue.queueLength = 0;
                }

                EnergyManager::ConsumeTransmissionEnergy(
                    g_states[i],
                    PACKET_SIZE_BITS,
                    WSNConfig::DEFAULT_TX_DISTANCE);

                g_states[i].queue.queueWaitingTime = 0.0;

                break;
            case 1: 
                SendPacket(i, sink1);

                if (g_states[i].queue.queueLength >= 2)
                {
                    g_states[i].queue.queueLength -= 2;
                }
                else
                {
                    g_states[i].queue.queueLength = 0;
                }

                EnergyManager::ConsumeTransmissionEnergy(
                    g_states[i],
                    PACKET_SIZE_BITS,
                    WSNConfig::DEFAULT_TX_DISTANCE);

                g_states[i].queue.queueWaitingTime = 0.0;

                break;
            case 2: 
                SendPacket(i, sink2);

                if (g_states[i].queue.queueLength >= 2)
                {
                    g_states[i].queue.queueLength -= 2;
                }
                else
                {
                    g_states[i].queue.queueLength = 0;
                }

                EnergyManager::ConsumeTransmissionEnergy(
                    g_states[i],
                    PACKET_SIZE_BITS,
                    WSNConfig::DEFAULT_TX_DISTANCE);

                g_states[i].queue.queueWaitingTime = 0.0;

                break;
            case 3: 
                EnergyManager::ConsumeReceptionEnergy(g_states[i], PACKET_SIZE_BITS); break; // Intentional buffering wait state
            case 4: 
                if (g_states[i].queue.queueLength > 0)
                {
                    g_states[i].queue.queueLength--;
                } 
            break; // Memory overflow dropping action
        }
    }

    // Guard safety bounds
    if(!EnergyManager::IsNodeAlive(g_states[i]))
    {
        g_states[i].isAlive=false;
    }

    // Environmental state update loop for all nodes
    for (uint32_t n = 0; n < NUM_NODES; n++) {
        g_states[n].queue.queueLength++;
        
        // Accumulate age tracking if packets wait inside the buffer
        if (g_states[n].queue.queueLength > 0) {
            g_states[n].queue.queueWaitingTime += WSNConfig::STEP_INTERVAL;
        } else {
            g_states[n].queue.queueWaitingTime = 0.0;
        }
        
        EnergyManager::ConsumeReceptionEnergy(g_states[n], PACKET_SIZE_BITS);
        if(!EnergyManager::IsNodeAlive(g_states[n]))
        {
            g_states[n].isAlive=false;
        }
        g_states[n].sensor.currentValue = WSNConfig::SENSOR_MIN + (rand() % static_cast<int>( WSNConfig::SENSOR_MAX - WSNConfig::SENSOR_MIN));
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
    
    lrWpanHelper.EnablePcapAll (std::string("scratch/pcaps/wsn-dqn"), true);
    
    g_gym = CreateObject<OpenGymInterface> (openGymPort);
    g_gym->SetGetObservationSpaceCb (MakeCallback (&GetObservationSpace));
    g_gym->SetGetActionSpaceCb (MakeCallback (&GetActionSpace));
    g_gym->SetGetObservationCb (MakeCallback (&GetObservation));
    g_gym->SetGetRewardCb (MakeCallback (&GetReward));
    g_gym->SetGetGameOverCb (MakeCallback (&GetGameOver));
    g_gym->SetGetExtraInfoCb (MakeCallback (&GetExtraInfo));
    g_gym->SetExecuteActionsCb (MakeCallback (&ExecuteActions));
    
    for (uint32_t i = 0; i < NUM_NODES; i++)
    {
        g_states[i].nodeId = i;

        g_states[i].energy.initialEnergy = WSNConfig::INITIAL_ENERGY;

        g_states[i].energy.residualEnergy = WSNConfig::INITIAL_ENERGY;

        g_states[i].energy.consumedEnergy = 0.0;

        g_states[i].sensor.currentValue = 0.0;

        g_states[i].sensor.dynamicThreshold = WSNConfig::INITIAL_THRESHOLD;

        g_states[i].queue.queueLength = 2;

        g_states[i].queue.queueWaitingTime = 0.0;

        g_states[i].isAlive = true;
    }
    
    Simulator::Schedule (MilliSeconds (100), &ScheduleNextStep, 100);
    Simulator::Stop (Seconds (60.0));
    Simulator::Run ();
    g_gym->NotifySimulationEnd ();
    Simulator::Destroy ();
    return 0;
}
