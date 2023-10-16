/*
 * This script simulates a complex scenario with multiple gateways and end
 * devices. The metric of interest for this script is the throughput of the
 * network.
 */

#include "ns3/end-device-lora-phy.h"
#include "ns3/gateway-lora-phy.h"
#include "ns3/class-a-end-device-lorawan-mac.h"
#include "ns3/gateway-lorawan-mac.h"
#include "ns3/simulator.h"
#include "ns3/log.h"
#include "ns3/pointer.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/lora-helper.h"
#include "ns3/node-container.h"
#include "ns3/mobility-helper.h"
#include "ns3/position-allocator.h"
#include "ns3/double.h"
#include "ns3/random-variable-stream.h"
#include "ns3/periodic-sender-helper.h"
#include "ns3/one-shot-sender.h"
#include "ns3/command-line.h"
#include "ns3/network-server-helper.h"
#include "ns3/correlated-shadowing-propagation-loss-model.h"
#include "ns3/building-penetration-loss.h"
#include "ns3/building-allocator.h"
#include "ns3/buildings-helper.h"
#include "ns3/forwarder-helper.h"
#include "ns3/okumura-hata-propagation-loss-model.h"
#include "ns3/basic-energy-source-helper.h"
#include "ns3/lora-radio-energy-model-helper.h"
#include "ns3/file-helper.h"
#include "ns3/names.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

using namespace ns3;
using namespace lorawan;

NS_LOG_COMPONENT_DEFINE ("ComplexLorawanNetworkExample");

//////////////////////
// Global Variables //
//////////////////////
// Network settings
int nDevices = 2000;
int nGateways = 1;
double radius = 10000; //Note that due to model updates, 7500 m is no longer the maximum distance 
double simulationTime = nDevices;
double lambda = 1; // Traffic Load
int nRun = 0;
std::string clusteringModel = "kmeans";

// Channel model
int realisticChannelModel = 2;
std::string models[] = {"baseline", "shadowing", "buildings"};
int appPeriodSeconds = 4;

// Output control
bool print = true;

// Files
// Files
std::string edPos = "";
std::string edOutputFile = "";
std::string gwPos = "";
std::string baseDir = "/home/thiago/Documentos/Doutorado/Simuladores/ns-3-dev/scratch/dap_clustering/";

// Traffic
int traffic = 0;
std::string trafficStrs[] = {"unconfirmed", "confirmed", "mixed"};

/////////////////////////
// Auxiliary Functions //
/////////////////////////
std::string 
Format(double v, int digits=4) {
    std::ostringstream ss;
    ss.precision(digits);
    ss << v;
    return ss.str();
}

void 
WriteInLog (std::string output)
{
  std::string fileName = baseDir + "data/" + clusteringModel + "/" + "tracker_" + Format(lambda, 2) + "_" 
                        + trafficStrs[traffic] + "_" + models[realisticChannelModel] + std::to_string(nGateways) 
                        + "gw.csv";
  std::ofstream ofs;
  ofs.open (fileName, std::ofstream::out | std::ofstream::app);
  if (ofs.is_open ())
    {
      //sent,received,pdr,rssi,snr,delay
      ofs << output << (clusteringModel == "tests" ? "," + std::to_string ((int)radius) : "") << std::endl;
      ofs.close ();
    }
}

std::vector<std::string> 
split (std::string str, const char del=',')
{
  std::vector<std::string> v;
  std::string temp = ""; 
  for(size_t i = 0; i < str.size(); i++)
    {
      if(str[i] != del)
        {
          temp += str[i];
        }
      else
        {
          v.push_back (temp);
          temp = "";
        }
    }
    v.push_back (temp);

  return v;
}

void 
SetPositionGWs (NodeContainer gateways)
{
  std::ifstream ifs;
  ifs.open (baseDir + gwPos, std::ifstream::in);
  if (!ifs.is_open ())
    return;

  MobilityHelper mobility;
  mobility.SetPositionAllocator ("ns3::ConstantPositionMobilityModel");
  Ptr<ListPositionAllocator> allocator = CreateObject<ListPositionAllocator> ();
  std::string line;
  for (int i = 0; std::getline(ifs, line); i++) 
    {
      std::vector<std::string> posStr = split (line);
      Ptr<Node> node = gateways.Get (i);      
      double x = std::stod (posStr[0]), y = std::stod (posStr[1]), z = 15.0;
      allocator->Add (Vector3D (x, y, z));
    }
  mobility.SetPositionAllocator (allocator);
  mobility.Install (gateways);
  
  ifs.close ();
}

void 
SetPositionEDs (NodeContainer endDevices)
{
  std::ifstream ifs;
  ifs.open (baseDir + edPos, std::ifstream::in);
  if (!ifs.is_open ())
    return;

  MobilityHelper mobility;
  mobility.SetPositionAllocator ("ns3::ConstantPositionMobilityModel");
  Ptr<ListPositionAllocator> allocator = CreateObject<ListPositionAllocator> ();
  std::string line;
  for (int i = 0; std::getline(ifs, line); i++) 
    {
      std::vector<std::string> posStr = split (line);
      Ptr<Node> node = endDevices.Get (i);      
      double x = std::stod (posStr[0]), y = std::stod (posStr[1]), z = 1.2;
      allocator->Add (Vector3D (x, y, z));
    }
  mobility.SetPositionAllocator (allocator);
  mobility.Install (endDevices);
  
  ifs.close ();
}

//////////
// Main //
//////////
int
main (int argc, char *argv[])
{
  CommandLine cmd;
  cmd.AddValue ("nDevices", "Number of end devices to include in the simulation", nDevices);
  cmd.AddValue ("nGateways", "Number of gateways to include in the simulation", nGateways);
  cmd.AddValue ("model", "0: Baseline (only PathLoss), 1: 0+Shadowing, 2: 1+Buildings",
                realisticChannelModel);
  cmd.AddValue ("print", "Whether or not to print various informations", print);
  cmd.AddValue ("lambda", "Load Traffic in pts/s", lambda);
  cmd.AddValue ("nRun", "Number of running", nRun);
  cmd.AddValue ("traffic", "Type of Traffic: {0: 'uncofirmed', 1: 'confirmed', 2: 'mixed'}", traffic);
  cmd.AddValue ("edPos", "Input File storing End Device datas", edPos);
  cmd.AddValue ("edOutputFile", "Output File to store End Device final datas", edOutputFile);
  cmd.AddValue ("gwPos", "File with the gateway positions", gwPos);
  cmd.AddValue ("nGateways", "Number of gateways", nGateways);
  cmd.AddValue ("cModel", "Clustering Model", clusteringModel);
  cmd.AddValue ("radius", "Size of the radius of the gateway cell", radius);
  cmd.Parse (argc, argv);

  if(nRun > 0)
    RngSeedManager::SetRun(nRun);    

  simulationTime = nDevices;

  appPeriodSeconds = simulationTime / lambda;
  //simulationTime = simulationTime / lambda * nSimulation;
  simulationTime = simulationTime / lambda * 1;

  std::cout << "Lambda=" << lambda << "," << "Simulation Time=" << simulationTime << "," << "App Period=" 
            << appPeriodSeconds << "," << "Number of Gateways=" << nGateways << std::endl;

  // Set up logging
  // LogComponentEnable ("ComplexLorawanNetworkExample", LOG_LEVEL_ALL);

  /***********
   *  Setup  *
   ***********/

  // Create the time value from the period
  Time appPeriod = Seconds (appPeriodSeconds);

  // Mobility
  //MobilityHelper mobility;
  /*mobility.SetPositionAllocator ("ns3::UniformDiscPositionAllocator", "rho", DoubleValue (radius),
                                 "X", DoubleValue (0.0), "Y", DoubleValue (0.0));
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");*/

  /************************
   *  Create the channel  *
   ************************/

  // Create the lora channel object
  /*Ptr<LogDistancePropagationLossModel> loss = CreateObject<LogDistancePropagationLossModel> ();
  loss->SetPathLossExponent (3.76);
  loss->SetReference (1, 7.7);*/
  Ptr<OkumuraHataPropagationLossModel> loss = CreateObject<OkumuraHataPropagationLossModel> ();
  loss->SetAttribute ("Frequency", DoubleValue (868000000));
  loss->SetAttribute ("Environment", EnumValue (UrbanEnvironment));
  loss->SetAttribute ("CitySize", EnumValue (MediumCity));

  if (realisticChannelModel == 1 || realisticChannelModel == 2)
    {
      // Create the correlated shadowing component
      Ptr<CorrelatedShadowingPropagationLossModel> shadowing =
          CreateObject<CorrelatedShadowingPropagationLossModel> ();

      // Aggregate shadowing to the logdistance loss
      loss->SetNext (shadowing);

      if (realisticChannelModel == 2)
        {
          // Add the effect to the channel propagation loss
          Ptr<BuildingPenetrationLoss> buildingLoss = CreateObject<BuildingPenetrationLoss> ();

          shadowing->SetNext (buildingLoss);
        }
    }

  Ptr<PropagationDelayModel> delay = CreateObject<ConstantSpeedPropagationDelayModel> ();

  Ptr<LoraChannel> channel = CreateObject<LoraChannel> (loss, delay);

  /************************
   *  Create the helpers  *
   ************************/

  // Create the LoraPhyHelper
  LoraPhyHelper phyHelper = LoraPhyHelper ();
  phyHelper.SetChannel (channel);

  // Create the LorawanMacHelper
  LorawanMacHelper macHelper = LorawanMacHelper ();

  // Create the LoraHelper
  LoraHelper helper = LoraHelper ();
  helper.EnablePacketTracking (); // Output filename
  // helper.EnableSimulationTimePrinting ();

  //Create the NetworkServerHelper
  NetworkServerHelper nsHelper = NetworkServerHelper ();

  //Create the ForwarderHelper
  ForwarderHelper forHelper = ForwarderHelper ();

  /************************
   *  Create End Devices  *
   ************************/

  // Create a set of nodes
  NodeContainer endDevices;
  endDevices.Create (nDevices);

  // Assign a mobility model to each node
  /*mobility.Install (endDevices);

  // Make it so that nodes are at a certain height > 0
  for (NodeContainer::Iterator j = endDevices.Begin (); j != endDevices.End (); ++j)
    {
      Ptr<MobilityModel> mobility = (*j)->GetObject<MobilityModel> ();
      Vector position = mobility->GetPosition ();
      position.z = 1.2;
      mobility->SetPosition (position);
    }*/
  
  SetPositionEDs(endDevices);
  

  // Create the LoraNetDevices of the end devices
  uint8_t nwkId = 54;
  uint32_t nwkAddr = 1864;
  Ptr<LoraDeviceAddressGenerator> addrGen =
      CreateObject<LoraDeviceAddressGenerator> (nwkId, nwkAddr);

  // Create the LoraNetDevices of the end devices
  macHelper.SetAddressGenerator (addrGen);
  phyHelper.SetDeviceType (LoraPhyHelper::ED);
  macHelper.SetDeviceType (LorawanMacHelper::ED_A);
  NetDeviceContainer endDevicesNetDevices = helper.Install (phyHelper, macHelper, endDevices);

  // Now end devices are connected to the channel

  // Connect trace sources
  if (traffic == 1)
    {
      for (NodeContainer::Iterator j = endDevices.Begin (); j != endDevices.End (); ++j)
        {
          Ptr<Node> node = *j;
          Ptr<LoraNetDevice> loraNetDevice = node->GetDevice (0)->GetObject<LoraNetDevice> ();
          Ptr<EndDeviceLorawanMac> mac = loraNetDevice->GetMac ()->GetObject<EndDeviceLorawanMac> ();;
          mac->SetMType (LorawanMacHeader::CONFIRMED_DATA_UP);
        }
    }

  /*********************
   *  Create Gateways  *
   *********************/

  // Create the gateway nodes (allocate them uniformely on the disc)
  NodeContainer gateways;
  gateways.Create (nGateways);

  /*Ptr<ListPositionAllocator> allocator = CreateObject<ListPositionAllocator> ();
  // Make it so that nodes are at a certain height > 0
  allocator->Add (Vector (0, 0, 15.0));
  if (nGateways == 2)
    {
      allocator->Add (Vector (radius, radius, 15.0));
      //allocator->Add (Vector (radius/2, radius/2, 15.0));
    }
  mobility.SetPositionAllocator (allocator);
  mobility.Install (gateways);*/
  SetPositionGWs(gateways);

  // Create a netdevice for each gateway
  phyHelper.SetDeviceType (LoraPhyHelper::GW);
  macHelper.SetDeviceType (LorawanMacHelper::GW);
  helper.Install (phyHelper, macHelper, gateways);

  /**********************
   *  Handle buildings  *
   **********************/

  double xLength = 300;
  double deltaX = 300;
  double yLength = 300;
  double deltaY = 300;
  int gridWidth = 2 * radius / (xLength + deltaX);
  int gridHeight = 2 * radius / (yLength + deltaY);
  if (realisticChannelModel == 0 || realisticChannelModel == 1)
    {
      gridWidth = 0;
      gridHeight = 0;
    }
  Ptr<GridBuildingAllocator> gridBuildingAllocator;
  gridBuildingAllocator = CreateObject<GridBuildingAllocator> ();
  gridBuildingAllocator->SetAttribute ("GridWidth", UintegerValue (gridWidth));
  gridBuildingAllocator->SetAttribute ("LengthX", DoubleValue (xLength));
  gridBuildingAllocator->SetAttribute ("LengthY", DoubleValue (yLength));
  gridBuildingAllocator->SetAttribute ("DeltaX", DoubleValue (deltaX));
  gridBuildingAllocator->SetAttribute ("DeltaY", DoubleValue (deltaY));
  gridBuildingAllocator->SetAttribute ("Height", DoubleValue (6));
  gridBuildingAllocator->SetBuildingAttribute ("NRoomsX", UintegerValue (2));
  gridBuildingAllocator->SetBuildingAttribute ("NRoomsY", UintegerValue (4));
  gridBuildingAllocator->SetBuildingAttribute ("NFloors", UintegerValue (2));
  gridBuildingAllocator->SetAttribute (
      "MinX", DoubleValue (-gridWidth * (xLength + deltaX) / 2 + deltaX / 2));
  gridBuildingAllocator->SetAttribute (
      "MinY", DoubleValue (-gridHeight * (yLength + deltaY) / 2 + deltaY / 2));
  BuildingContainer bContainer = gridBuildingAllocator->Create (gridWidth * gridHeight);

  // std::cout << bContainer.GetN () << std::endl;

  BuildingsHelper::Install (endDevices);
  BuildingsHelper::Install (gateways);

  // Print the buildings
  if (print)
    {
      std::ofstream myfile;
      myfile.open ("buildings.txt");
      std::vector<Ptr<Building>>::const_iterator it;
      int j = 1;
      for (it = bContainer.Begin (); it != bContainer.End (); ++it, ++j)
        {
          Box boundaries = (*it)->GetBoundaries ();
          myfile << "set object " << j << " rect from " << boundaries.xMin << "," << boundaries.yMin
                 << " to " << boundaries.xMax << "," << boundaries.yMax << std::endl;
        }
      myfile.close ();
    }

  /**********************************************
   *  Set up the end device's spreading factor  *
   **********************************************/
  if (clusteringModel != "tests")
    {
      macHelper.SetSpreadingFactorsUp (endDevices, gateways, channel);
    }
  else
    {
      Ptr<NetDevice> netDevice = endDevices.Get (0)->GetDevice (0);
      Ptr<LoraNetDevice> loraNetDevice = netDevice->GetObject<LoraNetDevice> ();
      Ptr<ClassAEndDeviceLorawanMac> mac =
          loraNetDevice->GetMac ()->GetObject<ClassAEndDeviceLorawanMac> ();
      mac->SetDataRate (0);
    }

  NS_LOG_DEBUG ("Completed configuration");

  /*********************************************
   *  Install applications on the end devices  *
   *********************************************/
  Time appStopTime = Seconds (simulationTime);
  PeriodicSenderHelper appHelper = PeriodicSenderHelper ();
  appHelper.SetPeriod (appPeriod);
  appHelper.SetPacketSize (31);
  /*Ptr<RandomVariableStream> rv = CreateObjectWithAttributes<UniformRandomVariable> (
      "Min", DoubleValue (0), "Max", DoubleValue (10));
  appHelper.SetPacketSizeRandomVariable (rv);*/
  ApplicationContainer appContainer = appHelper.Install (endDevices);

  appContainer.Start (Seconds (0));
  appContainer.Stop (appStopTime);

  /************************
   * Install Energy Model *
   ************************/

  BasicEnergySourceHelper basicSourceHelper;
  LoraRadioEnergyModelHelper radioEnergyHelper;

  // configure energy source
  basicSourceHelper.Set ("BasicEnergySourceInitialEnergyJ", DoubleValue (10000)); // Energy in J
  basicSourceHelper.Set ("BasicEnergySupplyVoltageV", DoubleValue (3.3));

  radioEnergyHelper.Set ("StandbyCurrentA", DoubleValue (0.0014));
  radioEnergyHelper.Set ("TxCurrentA", DoubleValue (0.028));
  radioEnergyHelper.Set ("SleepCurrentA", DoubleValue (0.0000015));
  radioEnergyHelper.Set ("RxCurrentA", DoubleValue (0.0112));

  radioEnergyHelper.SetTxCurrentModel ("ns3::LinearLoraTxCurrentModel");

  // install source on EDs' nodes
  EnergySourceContainer sources = basicSourceHelper.Install (endDevices);
  Names::Add ("/Names/EnergySource", sources.Get (0));

  // install device model
  DeviceEnergyModelContainer deviceModels = radioEnergyHelper.Install
      (endDevicesNetDevices, sources);
  
  // Get Output
  FileHelper fileHelper;
  std::string batteryFile = baseDir + "/data/" + clusteringModel + "/nRun_" + std::to_string (nRun) + "_" 
    + std::to_string(nGateways) + "gws_battery-level";
  fileHelper.ConfigureFile (batteryFile, FileAggregator::COMMA_SEPARATED);
  fileHelper.WriteProbe ("ns3::DoubleProbe", "/Names/EnergySource/RemainingEnergy", "Output");

  /**************************
   *  Create Network Server  *
   ***************************/

  // Create the NS node
  NodeContainer networkServer;
  networkServer.Create (1);

  // Create a NS for the network
  nsHelper.SetEndDevices (endDevices);
  nsHelper.SetGateways (gateways);
  nsHelper.Install (networkServer);

  //Create a forwarder for each gateway
  forHelper.Install (gateways);

  //////////
  // Logs //
  //////////
  if (clusteringModel == "tests")
    {
      std::string phyName = baseDir + "data/" + clusteringModel + "/nRun_" + std::to_string (nRun) + "_phy_" 
        + std::to_string (nGateways) + "gw.csv";
      helper.EnablePeriodicPhyPerformancePrinting (gateways, phyName, Time (simulationTime));
    }

  ////////////////
  // Simulation //
  ////////////////

  Simulator::Stop (appStopTime + Hours (1));

  NS_LOG_INFO ("Running simulation...");
  auto startTime = std::chrono::high_resolution_clock::now();

  Simulator::Run ();

  Simulator::Destroy ();

  ///////////////////////////
  // Print results to file //
  ///////////////////////////
  NS_LOG_INFO ("Computing performance metrics...");

  LoraPacketTracker& tracker = helper.GetPacketTracker ();
  std::string output = tracker.CountMacPacketsGlobally(Seconds (0), appStopTime);
  std::cout << output << std::endl;
  WriteInLog (output);

  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
  std::cout << "Tempo de execução: " << duration.count() / 1000 << " segundos" << std::endl;

  return 0;
}