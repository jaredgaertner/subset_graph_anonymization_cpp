#include <ctime>
#include <iostream>
#include <iterator>
#include <list>
#include <algorithm>
#include <limits>
#include <string>
#include <fstream>

#include <boost/algorithm/string.hpp>
#include <boost/assign/list_of.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/posix_time/posix_time_io.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/tee.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/copy.hpp>
#include <boost/graph/subgraph.hpp>
//#include <boost/graph/graphml.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/plod_generator.hpp>
#include <boost/graph/small_world_generator.hpp>
#include <boost/graph/max_cardinality_matching.hpp>
#include <boost/random.hpp>
#include <boost/timer.hpp>

#include "max_cardinality_matching_no_mapping.hpp"

// Create random function which outputs from [0,1)
boost::random::mt19937 rng( static_cast<unsigned int>(time(0)) );
static boost::random::uniform_01<boost::random::mt19937> uni01(rng);

// Graph definitions
struct vertex_info {};
typedef boost::property<boost::vertex_index_t, size_t> vertex_prop;

struct edge_info {};
typedef boost::property<boost::edge_index_t, size_t> edge_prop;

struct graph_info {};
typedef boost::property<boost::graph_name_t, std::string> graph_prop;

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, vertex_prop, edge_prop, graph_prop> Graph;
typedef boost::subgraph<Graph> SubGraph;
typedef boost::small_world_iterator<boost::random::minstd_rand, SubGraph> SmallWorldGenerator;
typedef boost::plod_iterator<boost::random::minstd_rand, SubGraph> PLODGenerator;

typedef boost::graph_traits<Graph>::vertex_descriptor vertex_descriptor_t;
typedef boost::graph_traits<Graph>::vertex_iterator vertex_iterator_t;
typedef boost::graph_traits<Graph>::edge_descriptor edge_descriptor_t;
typedef boost::graph_traits<Graph>::edge_iterator edge_iterator_t;
typedef boost::graph_traits<Graph>::out_edge_iterator out_edge_iterator_t;
typedef boost::graph_traits<Graph>::adjacency_iterator adjacency_iterator_t;
typedef std::pair<vertex_descriptor_t, vertex_descriptor_t> vertex_pair_t;

// Graph input choices
enum { EXIT, SMALL_WORLD_GRAPH, PLOD_GRAPH, ENRON_GRAPH, KARATE_GRAPH, POWERGRID_GRAPH, WIKI_GRAPH, EPINIONS_GRAPH };
std::string graph_titles [] = {"Exit", "Small-world Graph", "Power Law Out Degree Graph", "Enron Dataset Graph", "Karate Dataset Graph", "Powergrid Graph", "Wikipedia Vote Graph", "Epinions Graph"};

// Default graph/problem values
size_t default_number_of_experiments = 1;
size_t default_number_of_vertices = 1000;
size_t default_k = 50;
double default_subset_X_percent = 0.30;
size_t default_k_nearest_neighbors = 50;
size_t default_input_graph = SMALL_WORLD_GRAPH;
double default_alpha = 2.5;
double default_beta = 1000.0;

// Degree sequence definition
typedef std::pair<size_t,vertex_descriptor_t> degree_vertex_pair;

// Compare for sort creating a descending order
bool compare_descending(degree_vertex_pair i, degree_vertex_pair j){ return i.first > j.first; }

// Output to std::cout and file log.txt at same time using boost tee
typedef boost::iostreams::tee_device< std::ostream, boost::iostreams::stream<boost::iostreams::file_sink> > tee_device;
typedef boost::iostreams::stream<tee_device> tee_stream;
boost::iostreams::stream<boost::iostreams::file_sink> log_file_verbose, log_file_pertinent, experiment_data, experiment_data_averages;
tee_device tee(std::cout, log_file_pertinent);
tee_stream cout_and_log_file_pertinent(tee);

// Functions
void exit_program();
void select_input_graph(size_t& input_graph);
//void get_inputs(size_t& k, double& subset_X_percent);
//void get_inputs(double& alpha, double& beta);
//void get_inputs(size_t& number_of_experiments, size_t& k, double& subset_X_percent, size_t& number_of_vertices, size_t& k_nearest_neighbors);
//void get_inputs(size_t& number_of_experiments, size_t& k, double& subset_X_percent, size_t& number_of_vertices, double& alpha, double& beta);
template<class input_type>
void get_input_list(const std::string& description, std::vector<input_type>& input_list);
template<class input_type>
void get_input(const std::string& description, input_type& input_list);
void print_augmenting_path(const std::string& description, std::vector<vertex_descriptor_t> augmenting_path);
void print_degree_sequence(const std::string& description, std::vector<size_t> d);
void print_degree_sequence(const std::string& description, std::vector<degree_vertex_pair> d, bool verbose = true);
void print_degree_sequence(const std::string& description, std::map<vertex_descriptor_t, ptrdiff_t> d);
bool is_k_degree_anonymous(std::vector<degree_vertex_pair> degree_sequence, size_t k);
size_t ErdosGallaiThm(std::vector< std::vector<size_t> > SetOfDegreeSequences, size_t n);
size_t NumberOfKGroupings(size_t total, std::vector<size_t> d_reverse);
size_t DAGroupCost(std::vector<size_t> d, size_t start, size_t end);
std::vector< std::vector<degree_vertex_pair> > DegreeAnonymization(std::vector<degree_vertex_pair> d, size_t number_of_vertices, size_t k, bool AllPossible);
std::vector< vertex_pair_t > upper_degree_constrained_subgraph(Graph G, std::vector<degree_vertex_pair> d, std::map<vertex_descriptor_t, ptrdiff_t> upper_bounds, std::map<vertex_descriptor_t, ptrdiff_t> delta, size_t number_of_vertices_G_prime, size_t number_of_edges_G_prime, size_t number_of_vertices_G_prime_actual, size_t number_of_edges_G_prime_actual);

// Matching functions
size_t greedy_implicit_initial_matching(Graph G_prime, std::map<vertex_descriptor_t, std::set<vertex_descriptor_t> > &mates, std::map<vertex_descriptor_t, ptrdiff_t> &num_exposed_subvertices);
size_t extra_greedy_implicit_initial_matching(Graph G_prime, std::map<vertex_descriptor_t, std::set<vertex_descriptor_t> > &mates, std::map<vertex_descriptor_t, ptrdiff_t> &num_exposed_subvertices);
std::vector<vertex_descriptor_t> edmonds_implicit_matching(Graph G_prime, std::map<vertex_descriptor_t, std::set<vertex_descriptor_t> > mates, std::map<vertex_descriptor_t, ptrdiff_t> num_exposed_subvertices);
std::vector<vertex_descriptor_t> edmonds_find_augmenting_path(vertex_descriptor_t u, vertex_descriptor_t v, std::vector<vertex_descriptor_t> parent);

// Debug variables/functions
boost::timer t;		// Main level debug boost::timer
boost::timer t_l2;	// Level 2 debug boost::timer (debug statements contained within main level debug statements)
size_t recursion_level = 1;
void DEBUG_START(std::string);
void DEBUG_END(std::string);
void DEBUG_START_L2(std::string);
void DEBUG_END_L2(std::string);
void DEBUG_RECURSION_START(std::string);
void DEBUG_RECURSION_END(std::string);

int main()
{
	size_t number_of_experiments = 1;
	size_t number_of_vertices = 0;
	size_t k = 0;
	size_t k_nearest_neighbors = 0;
	double rewire_probability = 0.03; // uni01();
	size_t input_graph = SMALL_WORLD_GRAPH;
	double alpha = 0.0, beta = 0.0;
	double subset_X_percent = 0;

	//while(1){

		// Select type of graph
		select_input_graph(input_graph);
		
		// Not using user input, just cycling through 100 different cases, each with 100 experiments
		//		1. |V| in {100, 1000, 10000, 100000}
		//		2. k in {5, 15, 25, 35, 45}
		//		3. |X| in {.2|V|, .35|V|, .5|V|, .65|V|, .8|V|}

		std::vector<size_t> number_of_vertices_cases;
		std::vector<size_t> k_cases; //= assign::list_of(5)(10)(15)(20)(25);//(50)(100);
		std::vector<double> subset_X_percent_cases; //= assign::list_of(0.20)(0.35)(0.5)(0.65)(0.8);
		//std::vector<size_t> k_nearest_neighbors_cases; // = assign::list_of(10)(20)(30);

		///////////////////////////////////////////////////////////////////////////////////
		//
		// Get properties of G
		//
		// Get general inputs
		get_input<size_t>("number of experiments", number_of_experiments);
		get_input_list<size_t>("k", k_cases);
		get_input_list<double>("subset of X as a percent of G", subset_X_percent_cases);

		switch( input_graph ){
			case SMALL_WORLD_GRAPH:
				//get_inputs(number_of_experiments, k, subset_X_percent, number_of_vertices, k_nearest_neighbors);
				get_input<size_t>("k-nearest neighbors", k_nearest_neighbors);
				get_input<double>("probability of rewiring edges", rewire_probability);
				get_input_list<size_t>("number of vertices", number_of_vertices_cases);
				break;

			case PLOD_GRAPH:
				get_input<double>("alpha", alpha);
				get_input<double>("beta", beta);
				//get_inputs(number_of_experiments, k, subset_X_percent, number_of_vertices, alpha, beta);
				get_input_list<size_t>("number of vertices", number_of_vertices_cases);
				break;

			case ENRON_GRAPH:
				number_of_vertices = 36692;
				number_of_vertices_cases.push_back(number_of_vertices);
				break;

			case KARATE_GRAPH:
				//get_inputs(k, subset_X_percent);
				number_of_vertices = 34;
				number_of_vertices_cases.push_back(number_of_vertices);
				break;

			case POWERGRID_GRAPH:
				//get_inputs(k, subset_X_percent);
				number_of_vertices = 4941;
				number_of_vertices_cases.push_back(number_of_vertices);
				break;

			case WIKI_GRAPH:
				number_of_vertices = 7115;
				number_of_vertices_cases.push_back(number_of_vertices);
				break;

			case EPINIONS_GRAPH:
				number_of_vertices = 75879;
				number_of_vertices_cases.push_back(number_of_vertices);
				break;

			default:
				log_file_verbose << "ERROR!!! No graph option selected?" << std::endl;
				exit_program();
		}

		// Name output file (use if only one log file wanted for all cases, use date to identify it)
		boost::posix_time::time_facet *facet = new boost::posix_time::time_facet("%Y-%b-%d_%Hh-%Mm-%Ss");
		std::stringstream ss;
		ss.str("");
		ss.imbue( std::locale(ss.getloc(), facet) );
		boost::posix_time::ptime current_date_and_time = boost::posix_time::second_clock::local_time();
		ss << "log_verbose_" << current_date_and_time << ".txt";
		
		std::string log_file_verbose_name(ss.str());
		if( log_file_verbose.is_open() ){
			log_file_verbose.close();
		}
		log_file_verbose.open(log_file_verbose_name);

		ss.str("");
		ss << "log_pertinent_" << current_date_and_time << ".txt";
		std::string log_file_pertinent_name(ss.str());
		if( log_file_pertinent.is_open() ){
			log_file_pertinent.close();
		}
		log_file_pertinent.open(log_file_pertinent_name);

		// Start experiment data files (comma-separated values of pertinent data)
		ss.str("");
		ss << "experiment_data_" << current_date_and_time << ".csv";
		std::string experiment_data_name(ss.str());
		if( experiment_data.is_open() ){
			experiment_data.close();
		}
		experiment_data.open(experiment_data_name);
		experiment_data << "ExpNo,|V|,k,|X|-pct,opt,success,Intra-ad,Extra-ad,time,sum_def,failed_vertices,failed_cost,|V_com|,|E_com|,|V_com'|,|E_com'|" << std::endl;


		ss.str("");
		ss << "experiment_data_averages_" << current_date_and_time << ".csv";
		std::string experiment_data_averages_name(ss.str());
		if( experiment_data_averages.is_open() ){
			experiment_data_averages.close();
		}
		experiment_data_averages.open(experiment_data_averages_name);
		experiment_data_averages << "num_of_exp,|V|,k,|X|-pct,opt,success,Intra-ad,Extra-ad,time,sum_def,failed_vertices,failed_cost,|V_com|,|E_com|,|V_com'|,|E_com'|" << std::endl;

		// Set all log files to print "true" for bool instead of 1
		log_file_verbose << std::boolalpha;
		log_file_pertinent << std::boolalpha;
		experiment_data << std::boolalpha;
		experiment_data_averages << std::boolalpha;

		std::vector<size_t>::iterator number_of_vertices_it, k_it;
		std::vector<double>::iterator subset_X_percent_it;
		for(number_of_vertices_it = number_of_vertices_cases.begin(); number_of_vertices_it < number_of_vertices_cases.end(); number_of_vertices_it++){
			number_of_vertices = *number_of_vertices_it;
			//k_nearest_neighbors = k_nearest_neighbors_cases.front();	k_nearest_neighbors_cases.erase( k_nearest_neighbors_cases.begin() );

			for(subset_X_percent_it = subset_X_percent_cases.begin(); subset_X_percent_it < subset_X_percent_cases.end(); subset_X_percent_it++){
				subset_X_percent = *subset_X_percent_it;
				for(k_it = k_cases.begin(); k_it < k_cases.end(); k_it++){
					k = *k_it;

					//// Name output file (use if different log files for each case wanted)
					////posix_time::time_facet *facet = new posix_time::time_facet("%Y-%b-%d_%Hh-%Mm-%Ss");
					//std::stringstream ss;
					std::stringstream current_graph_case;
					//ss.str("");
					current_graph_case.str("");
					current_graph_case << "(" << graph_titles[input_graph] << ")_n_" << number_of_vertices << "_k_" << k << "_subset_" << (int)(100.0 * subset_X_percent);
					////posix_time::ptime current_date_and_time = posix_time::second_clock::local_time();
					//ss << "log_" << current_graph_case.str() << "_verbose.txt";
					//
					//std::string log_file_verbose_name(ss.str());
					//if( log_file_verbose.is_open() ){
					//	log_file_verbose.close();
					//}
					//log_file_verbose.open(log_file_verbose_name);

					//ss.str("");
					//ss << "log_" << current_graph_case.str() << "_pertinent.txt";
					//std::string log_file_pertinent_name(ss.str());
					//if( log_file_pertinent.is_open() ){
					//	log_file_pertinent.close();
					//}
					//log_file_pertinent.open(log_file_pertinent_name);

					//// Set log_file_pertinent to print "true" for bool instead of 1
					//log_file_pertinent << boolalpha;

					cout_and_log_file_pertinent << std::endl << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
					cout_and_log_file_pertinent << "Start case:" << std::endl;
					cout_and_log_file_pertinent << "\tNumber of vertices: " << number_of_vertices << std::endl;
					cout_and_log_file_pertinent << "\tk: " << k << std::endl;
					//cout_and_log_file_pertinent << "\tk-nearest neighbors (Small-world graph property): " << k_nearest_neighbors << std::endl;
					cout_and_log_file_pertinent << "\tSize of X (percentage of number of vertices): " << subset_X_percent << std::endl;
					cout_and_log_file_pertinent << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;


					SubGraph G_orig(number_of_vertices);

					switch( input_graph ){
						case SMALL_WORLD_GRAPH:
							{
							///////////////////////////////////////////////////////////////////////////////////
							//
							// Create a small-world graph (http://www.boost.org/doc/libs/1_48_0/libs/graph/doc/small_world_generator.html)
							//	small_world_iterator(RandomGenerator& gen, vertices_size_type n,
							//                    vertices_size_type k, double probability,
							//                    bool allow_self_loops = false);
							//	Constructs a small-world generator iterator that creates a graph with n vertices, each connected to its k nearest neighbors. Probabiliboost::ties are drawn from the random number generator gen. Self-loops are permitted only when allow_self_loops is true.
							//
								
							DEBUG_START("Creating small-world graph ...");
							boost::random::minstd_rand gen;
							bool allow_self_loops = false;
							log_file_verbose << "Number of vertices: " << number_of_vertices << std::endl;
							log_file_verbose << "Each connected to its " << k_nearest_neighbors << " nearest neighbors." << std::endl;
							log_file_verbose << "Edges in the graph are randomly rewired to different vertices with a probability " << rewire_probability << " and self-loops are set to " << allow_self_loops << "." << std::endl;
							SmallWorldGenerator smg_it;
							for(smg_it = SmallWorldGenerator(gen, number_of_vertices, k_nearest_neighbors, rewire_probability, allow_self_loops); smg_it != SmallWorldGenerator(); ++smg_it){
								add_edge( (*smg_it).first, (*smg_it).second, G_orig );
							}
							log_file_verbose << "Number of edges: " << num_edges(G_orig) << std::endl;
							DEBUG_END("Creating small-world graph ...");

							DEBUG_START("Writing graph G to graph file in graphviz .dot format ...");
							ss.str("");
							ss << "G_" << current_graph_case.str() << ".gv";
							boost::iostreams::stream<boost::iostreams::file_sink> graph_file(ss.str());
							write_graphviz(graph_file, G_orig);
							DEBUG_END("Writing graph G to graph file in graphviz .dot format ...");
							}
							break;

						case PLOD_GRAPH:
							{
							///////////////////////////////////////////////////////////////////////////////////
							//
							// Create a power law out degree graph (http://www.boost.org/doc/libs/1_49_0/libs/graph/doc/plod_generator.html)
							//	plod_iterator();
							//		Constructs a past-the-end iterator. 
							//	plod_iterator(RandomGenerator& gen, vertices_size_type n,
							//              double alpha, double beta, bool allow_self_loops = false);
							//		Constructs a PLOD generator iterator that creates a graph with n vertices. Probabiliboost::ties are drawn from the random number generator gen. Self-loops are permitted only when allow_self_loops is true.
								
							DEBUG_START("Creating power law out degree graph ...");
							boost::random::minstd_rand gen;
							bool allow_self_loops = false;
							double probability = uni01();
							log_file_verbose << "Number of vertices: " << number_of_vertices << std::endl;
							log_file_verbose << "Alpha: " << alpha << std::endl;
							log_file_verbose << "Beta: " << beta << std::endl;
							log_file_verbose << "Self-loops are set to " << allow_self_loops << "." << std::endl;
							PLODGenerator plod_it;
							for(plod_it = PLODGenerator(gen, number_of_vertices, alpha, beta, allow_self_loops); plod_it != PLODGenerator(); ++plod_it){
								add_edge( (*plod_it).first, (*plod_it).second, G_orig );
							}
							log_file_verbose << "Number of edges: " << num_edges(G_orig) << std::endl;
							DEBUG_END("Creating power law out degree graph ...");

							DEBUG_START("Writing graph G to graph file in graphviz .dot format ...");
							ss.str("");
							ss << "G_" << current_graph_case.str() << ".gv";
							boost::iostreams::stream<boost::iostreams::file_sink> graph_file(ss.str());
							write_graphviz(graph_file, G_orig);
							DEBUG_END("Writing graph G to graph file in graphviz .dot format ...");
							}
							break;

						case KARATE_GRAPH:
							{
							// Karate graph properties
							size_t number_of_edges = 78;

							// Read in Karate data
							DEBUG_START("Reading in karate data ...");
							std::vector< std::pair<size_t,size_t> > edge_list( number_of_edges );
							std::string line;
							boost::iostreams::stream<boost::iostreams::file_source> input_file("karate.txt");
							size_t number_of_edges_read = 0;
							if(input_file.is_open()){
								while(getline(input_file, line)){
									if(line.at(0) != '#'){
										std::vector<std::string> tokens;
										boost::split(tokens, line, boost::is_any_of("\t"));
										size_t u = atoi(tokens.at(0).c_str());
										size_t v = atoi(tokens.at(1).c_str());
										edge_list[number_of_edges_read++] = std::make_pair(u, v);
									}
								}
								input_file.close();
							}
							else{
								cout_and_log_file_pertinent << "ERROR!!! Unable to open file." << std::endl;
								exit_program();
							}
							DEBUG_END("Reading in karate data ...");

							DEBUG_START("Creating graph from karate data ...");
							std::vector< std::pair<size_t,size_t> >::iterator eit;
							for(eit = edge_list.begin(); eit < edge_list.end(); ++eit){
								add_edge( (*eit).first, (*eit).second, G_orig );
							}
							DEBUG_END("Creating graph from karate data ...");

							//DEBUG_START("Writing graph G to graph file in graphviz .dot format ...");
							//ss.str("");
							//ss << "G_" << current_graph_case.str() << ".gv";
							//boost::iostreams::stream<boost::iostreams::file_sink> graph_file(ss.str());
							//write_graphviz(graph_file, G_orig);
							//DEBUG_END("Writing graph G to graph file in graphviz .dot format ...");
							}
							break;

						case POWERGRID_GRAPH:
							{
							size_t number_of_edges = 6594;

							// Read in Powergrid data
							DEBUG_START("Reading in powergrid data ...");
							std::vector< std::pair<size_t,size_t> > edge_list( number_of_edges );
							std::string line;
							boost::iostreams::stream<boost::iostreams::file_source> input_file("USpowerGrid.mtx");
							size_t number_of_edges_read = 0;
							if(input_file.is_open()){
								while(getline(input_file, line)){
									if(line.at(0) != '%'){
										std::vector<std::string> tokens;
										boost::split(tokens, line, boost::is_any_of(" "));
										size_t u = atoi(tokens.at(0).c_str());
										size_t v = atoi(tokens.at(1).c_str());
										edge_list[number_of_edges_read++] = std::make_pair(u, v);
										if( number_of_edges_read % 1000 == 0 ){
											log_file_verbose << "Read " << number_of_edges_read << " edges out of 6594." << std::endl;
										}
									}
								}
								input_file.close();
							}
							else{
								cout_and_log_file_pertinent << "ERROR!!! Unable to open file." << std::endl;
								exit_program();
							}
							DEBUG_END("Reading in powergrid data ...");

							DEBUG_START("Creating graph from powergrid data ...");
							std::vector< std::pair<size_t,size_t> >::iterator eit;
							for(eit = edge_list.begin(); eit < edge_list.end(); ++eit){
								add_edge( (*eit).first, (*eit).second, G_orig );
							}
							DEBUG_END("Creating graph from powergrid data ...");

							//DEBUG_START("Writing graph G to graph file in graphviz .dot format ...");
							//ss.str("");
							//ss << "G_" << current_graph_case.str() << ".gv";
							//boost::iostreams::stream<boost::iostreams::file_sink> graph_file(ss.str());
							//write_graphviz(graph_file, G_orig);
							//DEBUG_END("Writing graph G to graph file in graphviz .dot format ...");
							}
							break;

						case ENRON_GRAPH:
							{
							// Enron graph properties
							size_t number_of_edges = 367662;

							// Read in Enron data
							DEBUG_START("Reading in enron data ...");
							std::vector< std::pair<size_t,size_t> > edge_list( number_of_edges / 2);
							std::string line;
							boost::iostreams::stream<boost::iostreams::file_source> input_file("Email-Enron.txt");
							size_t number_of_edges_read = 0;
							if(input_file.is_open()){
								while(getline(input_file, line)){
									if(line.at(0) != '#'){
										std::vector<std::string> tokens;
										boost::split(tokens, line, boost::is_any_of("\t"));
										size_t u = atoi(tokens.at(0).c_str());
										size_t v = atoi(tokens.at(1).c_str());
										if( u < v ){
											edge_list[number_of_edges_read++] = std::make_pair(u, v);
											if( number_of_edges_read % 10000 == 0 ){
												log_file_verbose << "Read " << number_of_edges_read << " edges out of 183831." << std::endl;
											}
										}
									}
								}
								input_file.close();
							}
							else{
								cout_and_log_file_pertinent << "ERROR!!! Unable to open file." << std::endl;
								exit_program();
							}
							DEBUG_END("Reading in enron data ...");

							DEBUG_START("Creating graph from enron data ...");
							std::vector< std::pair<size_t,size_t> >::iterator eit;
							for(eit = edge_list.begin(); eit < edge_list.end(); ++eit){
								add_edge( (*eit).first, (*eit).second, G_orig );
							}
							DEBUG_END("Creating graph from enron data ...");

							// Write Enron data in graphml format
							//dynamic_properties dp;
							//boost::iostreams::stream<boost::iostreams::file_sink> output_file("Enron Email Data.graphml");
							//DEBUG_START("Writing enron data to file Enron Email Data.graphml ...");
							//write_graphml(output_file, G, dp);
							//DEBUG_END("Writing enron data to file Enron Email Data.graphml ...");
							//output_file.close();

							// Read in Enron data in graphml format
							//Graph G;
							//boost::iostreams::stream<boost::iostreams::file_source> input_file("Enron Email Data.graphml");
							//dynamic_properties dp;
							//property_std::map<Graph, vertex_index_t>::type node_id_map = get(vertex_index, G);
							//dp.property("node_id", node_id_map);
							//DEBUG_START("Reading in enron data ...");
							//read_graphml(input_file, G, dp);
							//DEBUG_END("Reading in enron data ...");

							//DEBUG_START("Writing graph G to graph file in graphviz .dot format ...");
							//ss.str("");
							//ss << "G_" << current_graph_case.str() << ".gv";
							//boost::iostreams::stream<boost::iostreams::file_sink> graph_file(ss.str());
							//write_graphviz(graph_file, G_orig);
							//DEBUG_END("Writing graph G to graph file in graphviz .dot format ...");
							}
							break;

						case WIKI_GRAPH:
							{
							size_t number_of_edges = 103689;

							// Read in wiki data, considered undirected (even though each vote is directed)
							DEBUG_START("Reading in wiki vote data ...");
							std::set< std::pair<size_t,size_t> > edge_list;
							std::string line;
							boost::iostreams::stream<boost::iostreams::file_source> input_file("Wiki-Vote.txt");
							size_t number_of_edges_read = 0;
							if(input_file.is_open()){
								while(getline(input_file, line)){
									if(line.at(0) != '#'){
										std::vector<std::string> tokens;
										boost::split(tokens, line, boost::is_any_of("\t"));
										size_t u = atoi(tokens.at(0).c_str());
										size_t v = atoi(tokens.at(1).c_str());

										if( u < v )
											edge_list.insert( std::make_pair(u, v) );	// Does not insert duplicates (by only inserting pair with lower value first into set)
										else
											edge_list.insert( std::make_pair(v, u) );

										number_of_edges_read++;
										if( number_of_edges_read % 10000 == 0 ){
											log_file_verbose << "Read " << number_of_edges_read << " edges out of " << number_of_edges << "." << std::endl;
										}
									}
								}
								input_file.close();
							}
							else{
								cout_and_log_file_pertinent << "ERROR!!! Unable to open file." << std::endl;
								exit_program();
							}
							DEBUG_END("Reading in wiki vote data ...");

							DEBUG_START("Creating graph from wiki vote data ...");
							std::set< std::pair<size_t,size_t> >::iterator eit;
							for(eit = edge_list.begin(); eit != edge_list.end(); ++eit){
								add_edge( (*eit).first, (*eit).second, G_orig );
							}
							log_file_verbose << "Number of edges: " << num_edges(G_orig) << std::endl;
							DEBUG_END("Creating graph from wiki vote data ...");

							//DEBUG_START("Writing graph G to graph file in graphviz .dot format ...");
							//ss.str("");
							//ss << "G_wiki.gv";
							//boost::iostreams::stream<boost::iostreams::file_sink> graph_file(ss.str());
							//write_graphviz(graph_file, G_orig);
							//DEBUG_END("Writing graph G to graph file in graphviz .dot format ...");
							}
							break;

						case EPINIONS_GRAPH:
							{
							size_t number_of_edges = 508837;

							// Read in epinions data, considered undirected (even though each vote is directed)
							DEBUG_START("Reading in epinions data ...");
							std::set< std::pair<size_t,size_t> > edge_list;
							std::string line;
							boost::iostreams::stream<boost::iostreams::file_source> input_file("soc-Epinions1.txt");
							size_t number_of_edges_read = 0;
							if(input_file.is_open()){
								while(getline(input_file, line)){
									if(line.at(0) != '#'){
										std::vector<std::string> tokens;
										boost::split(tokens, line, boost::is_any_of("\t"));
										size_t u = atoi(tokens.at(0).c_str());
										size_t v = atoi(tokens.at(1).c_str());

										if( u < v )
											edge_list.insert( std::make_pair(u, v) );	// Does not insert duplicates (by only inserting pair with lower value first into set)
										else
											edge_list.insert( std::make_pair(v, u) );

										number_of_edges_read++;
										if( number_of_edges_read % 10000 == 0 ){
											log_file_verbose << "Read " << number_of_edges_read << " edges out of " << number_of_edges << "." << std::endl;
										}
									}
								}
								input_file.close();
							}
							else{
								cout_and_log_file_pertinent << "ERROR!!! Unable to open file." << std::endl;
								exit_program();
							}
							DEBUG_END("Reading in epinions data ...");

							DEBUG_START("Creating graph from epinions data ...");
							std::set< std::pair<size_t,size_t> >::iterator eit;
							for(eit = edge_list.begin(); eit != edge_list.end(); ++eit){
								add_edge( (*eit).first, (*eit).second, G_orig );
							}
							log_file_verbose << "Number of edges: " << num_edges(G_orig) << std::endl;
							DEBUG_END("Creating graph from epinions data ...");

							//DEBUG_START("Writing graph G to graph file in graphviz .dot format ...");
							//ss.str("");
							//ss << "G_epinions.gv";
							//boost::iostreams::stream<boost::iostreams::file_sink> graph_file(ss.str());
							//write_graphviz(graph_file, G_orig);
							//DEBUG_END("Writing graph G to graph file in graphviz .dot format ...");
							}
							break;

						default:
							log_file_verbose << "ERROR!!! No graph option selected?" << std::endl;
							exit_program();
					}

					// Total values to keep track of through all experiments
					size_t optimal_graphs = 0;	// Number of graphs with only edge additions in X (or only 1 in G\X)
					size_t current_experiment = 1; // Keep track of the current experiment number
					size_t successes = 0, failures = 0; // Number of times the experiments have succeeded and failed
					size_t intra_X_num_edges_total = 0, non_intra_X_num_edges_total = 0;
					size_t number_failed_vertices_total = 0, anonymization_failed_cost_total = 0, anonymization_cost_total = 0;
					size_t number_of_vertices_X_com_total = 0, number_of_edges_X_com_total = 0;
					size_t number_of_vertices_X_prime_actual_total = 0, number_of_edges_X_prime_actual_total = 0;
					double experiment_time_in_seconds_total = 0.0;

					// Start looping until number_of_experiments have been run
					while( current_experiment <= number_of_experiments ){
						log_file_verbose << std::endl << std::endl;
						log_file_verbose << "------------------------------------------------------------------------------" << std::endl;
						log_file_verbose << "Experiment number: " << current_experiment << std::endl;
						log_file_verbose << "\tNumber of vertices: " << number_of_vertices << std::endl;
						log_file_verbose << "\tNumber of vertices (double check size in G): " << num_vertices(G_orig) << std::endl;
						log_file_verbose << "\tk: " << k << std::endl;
						log_file_verbose << "\tSize of X (percentage of number of vertices): " << subset_X_percent << std::endl;

						log_file_pertinent << std::endl << std::endl;
						log_file_pertinent << "------------------------------------------------------------------------------" << std::endl;
						log_file_pertinent << "Experiment number: " << current_experiment << std::endl;
						boost::timer experiment_time;
						experiment_time.restart();
						SubGraph G(G_orig);

						///////////////////////////////////////////////////////////////////////////////////
						//
						// Find a small subset of vertices, X, of G
						//
						DEBUG_START("Creating subset X of G ...");
						std::set<vertex_descriptor_t> X_vertices;
						if( subset_X_percent <= 0.99 ){
							while(X_vertices.size() < subset_X_percent * number_of_vertices){
								// randomly add vertices to X
								double rand1 = uni01();
								size_t add_vertex_num = (size_t)(rand1 * num_vertices(G));
								if( degree( vertex(add_vertex_num, G), G) > 0 ){
									X_vertices.insert( vertex(add_vertex_num, G) );
								}
							}

							//std::set<vertex_descriptor_t>::iterator sit;
							//log_file_verbose << "Vertices chosen for X from G: " << std::endl;
							//for(sit = X_vertices.begin(); sit != X_vertices.end(); ++sit){
							//	log_file_verbose << *sit << ",";
							//}
							//log_file_verbose << std::endl << std::endl;
							
							// Output vertices chosen for X to file
							
							ss.str("");
							ss << "X_k_" << k << "_percent_" << (size_t)(subset_X_percent * 100.0) << "_exp_" << current_experiment << ".txt";
							boost::iostreams::stream<boost::iostreams::file_sink> X_output_file(ss.str());
							std::set<vertex_descriptor_t>::iterator sit;
							for(sit = X_vertices.begin(); sit != X_vertices.end(); ++sit){
								X_output_file << *sit << std::endl;
							}
							X_output_file.close();
						}
						else{
							vertex_iterator_t ui, ui_end, vi;
							for(boost::tie(ui, ui_end) = vertices(G); ui != ui_end; ++ui){
								X_vertices.insert(*ui);
							}
						}

						SubGraph& X_sub = G.create_subgraph(X_vertices.begin(), X_vertices.end());
						Graph X;
						boost::copy_graph(X_sub, X);
						size_t number_of_vertices_X = num_vertices(X);
						log_file_verbose << "Number of vertices in X: " << number_of_vertices_X << std::endl;
						DEBUG_END("Creating subset X of G ...");

						//DEBUG_START("Writing graph X to graph file in graphviz .dot format ...");
						//ss.str("");
						//ss << "X_" << current_date_and_time << ".gv";
						//boost::iostreams::stream<boost::iostreams::file_sink> graph_file(ss.str());
						//write_graphviz(graph_file, X);
						//DEBUG_END("Writing graph X to graph file in graphviz .dot format ...");

						// Find degree sequence
						std::vector< degree_vertex_pair > d;
						vertex_iterator_t ui, ui_end, vi;
						for(boost::tie(ui, ui_end) = vertices(X); ui != ui_end; ++ui){
							//degree_vertex_pair degree_vertex( degree( *ui,X ), *ui ); // Anonymize induced X
							degree_vertex_pair degree_vertex( degree(X_sub.local_to_global(*ui), G), *ui ); // Anonymize X wrt G
							d.push_back(degree_vertex);
						}
						sort(d.begin(), d.end(), compare_descending);
						//print_degree_sequence("Degree sequence", d);
						
						// Check for clique
						if( d.front().first == d.back().first ){
							log_file_verbose << "Clique or no edges, skipping." << std::endl;
							log_file_pertinent << "Clique or no edges, skipping." << std::endl;
							continue;
						}

						/*
						// Determine total number of k-groupings
						size_t total = 0;
						std::vector<size_t> d_reverse(d.rbegin(), d.rend());
						total = NumberOfKGroupings(total, d_reverse);
						log_file_verbose << "The total number of k-groupings: " << total << std::endl;
						*/
						

						///////////////////////////////////////////////////////////////////////////////////
						//
						// Degree Anonymization (Lui and Terzi 2008)
						//
						DEBUG_START("Determining degree anonymization ...");
						std::vector< std::vector<degree_vertex_pair> > SetOfDegreeSequences;
						SetOfDegreeSequences = DegreeAnonymization(d, number_of_vertices_X, k, false);
						std::vector<degree_vertex_pair> AnonymizedDegreeSequence(SetOfDegreeSequences.back());
						
						//print_degree_sequence("Anonymized degree sequence", AnonymizedDegreeSequence);
						ss.str("");
						ss << "anonymized_degree_sequence_k_" << k << "_Xpercent_" << (size_t)(subset_X_percent * 100.0) << "_exp_" << current_experiment << ".txt";
						boost::iostreams::stream<boost::iostreams::file_sink> anonymized_degree_sequence_output_file(ss.str());
						log_file_verbose << "Anonymized degree sequence (vertex id below in brackets): " << std::endl;
						std::vector<degree_vertex_pair>::iterator it;
						for(it = AnonymizedDegreeSequence.begin(); it < AnonymizedDegreeSequence.end()-1; ++it){
							anonymized_degree_sequence_output_file << (*it).first << "\t";
						}
						anonymized_degree_sequence_output_file << (*it).first << std::endl;
						for(it = AnonymizedDegreeSequence.begin(); it < AnonymizedDegreeSequence.end()-1; ++it){
							anonymized_degree_sequence_output_file << "(" << (*it).second << ")" << "\t";
						}
						anonymized_degree_sequence_output_file << "(" << (*it).second << ")" << std::endl;
						anonymized_degree_sequence_output_file.close();

						DEBUG_END("Determining degree anonymization ...");

						// Check if it is a real degree sequence
						// ErdosGallaiThm(SetOfDegreeSequences.back(), number_of_vertices);

						///////////////////////////////////////////////////////////////////////////////////
						//
						// Find lower and upper bounds and delta
						//
						DEBUG_START("Determine the upper bounds for vertices in X");
						std::map<vertex_descriptor_t, ptrdiff_t> upper_bounds;
						std::vector<degree_vertex_pair>::iterator dvit;
						size_t anonymized_degree_sequence_index = 0;
						size_t number_of_vertices_X_prime = 0;
						size_t number_of_edges_X_prime = 0; // num_edges(X_com); // if including only internal to external edges, use 0, for all edges use num_edges(X_com)
						size_t anonymization_cost = 0;
						for(dvit = d.begin(); dvit != d.end(); ++dvit){
							vertex_descriptor_t u = (*dvit).second;
							size_t d_i = number_of_vertices_X - 1 - degree(u, X);
							ptrdiff_t u_i = AnonymizedDegreeSequence.at(anonymized_degree_sequence_index++).first - (*dvit).first;
							upper_bounds[u] = u_i;
							ptrdiff_t delta_i = d_i - u_i;
							anonymization_cost += u_i;

							// Determine the number of edges that would be used if vertices with u_i == 0 weren't excluded
							number_of_vertices_X_prime += d_i + delta_i;
							number_of_edges_X_prime += d_i * delta_i;
						}
						print_degree_sequence("Upper bounds", upper_bounds);
						DEBUG_END("Determine the upper bounds for vertices in X");


						// Find the complement of the subset X
						DEBUG_START("Finding complement of X ...");
						//size_t number_of_edges_read_X = 0;
						//std::vector< std::pair<vertex_descriptor_t,vertex_descriptor_t> > edge_list_X_com( number_of_vertices_X * (number_of_vertices_X - 1) / 2 - num_edges(X) );
						Graph X_com( number_of_vertices_X );
						for(boost::tie(ui, ui_end) = vertices(X); ui != ui_end; ++ui){
							for(vi = ui+1; vi != ui_end; ++vi){
								// Only add edge if not in X AND it's upper bounds of both vertices is greater than 0 (ignoring any edges that cannot be in matching)
								if( !edge(*ui, *vi, X).second && upper_bounds[*ui] > 0 && upper_bounds[*vi] > 0 ){
									//edge_list_X_com[number_of_edges_read_X++] = std::make_pair(*ui, *vi);
									add_edge(*ui,*vi,X_com);
								}
							}
						}
						//Graph X_com(edge_list_X_com.begin(), edge_list_X_com.end(), number_of_vertices_X);
						size_t number_of_vertices_X_com = 0;
						size_t number_of_edges_X_com = num_edges(X_com);
						DEBUG_END("Finding complement of X ...");

						DEBUG_START("Determine the lower bounds and delta values for every vertex in X/X_com");
						std::map<vertex_descriptor_t, ptrdiff_t> lower_bounds, delta;
						size_t number_of_vertices_X_prime_actual = 0, number_of_edges_X_prime_actual = 0;
						for(size_t i = 0; i < d.size(); i++){
							vertex_descriptor_t u = d.at(i).second;
							size_t d_i = degree(u, X_com);
							size_t u_i = upper_bounds[u];
							ptrdiff_t delta_i = d_i - u_i;
							if( delta_i < 0 ){
								delta_i = 0;
							}
							delta[u] = delta_i;

							vertex_descriptor_t global_index_u = X_sub.local_to_global(u);
							size_t degree_in_X = degree(u, X_sub);
							size_t degree_in_G = degree(global_index_u, G);
							size_t degree_in_only_G = degree_in_G - degree_in_X;
							ptrdiff_t l_i = upper_bounds[u] - degree_in_only_G;
							if( l_i < 0 ){
								l_i = 0;
							}
							lower_bounds[u] = l_i;

							// Determine number of vertices/edges total if we don't include vertices/edges where u_i == 0
							if( u_i > 0 ){
								number_of_vertices_X_com++;
								number_of_vertices_X_prime_actual += d_i + delta_i;
								number_of_edges_X_prime_actual += d_i * delta_i;
							}
						}
						number_of_edges_X_prime_actual += number_of_edges_X_com;

						//print_degree_sequence("Lower bounds", lower_bounds);
						//print_degree_sequence("Delta values", delta);
						DEBUG_END("Determine the upper, lower bounds and delta values for every vertex in X/X_com");

						///////////////////////////////////////////////////////////////////////////////////
						//
						// Create X_star
						//

						///////////////////////////////////////////////////////////////////////////////////
						//
						// Find upper degree-constrained subgraph (H_star) on X_star
						//
						std::vector< vertex_pair_t > H_star = upper_degree_constrained_subgraph(X_com, d, upper_bounds, delta, number_of_vertices_X_prime, number_of_edges_X_prime, number_of_vertices_X_prime_actual, number_of_edges_X_prime_actual);

						DEBUG_START("Adding edges from H to X, displaying final degree sequence ...");
						size_t intra_X_num_edges = 0;
						std::vector< std::pair<vertex_descriptor_t,vertex_descriptor_t> >::iterator vvit;
						log_file_verbose << "Edges to add to X to make it k-degree anonymous:" << std::endl << std::endl;
						log_file_verbose << "Global\t\tLocal" << std::endl;
						for(vvit = H_star.begin(); vvit < H_star.end(); ++vvit){
							log_file_verbose << "(" << X_sub.local_to_global( (*vvit).first ) << "," << X_sub.local_to_global( (*vvit).second ) << ")\t\t(" << (*vvit).first << "," << (*vvit).second << ")" << std::endl;
							add_edge( vertex((*vvit).first, X_sub), vertex((*vvit).second, X_sub), X_sub); // Add edge to X
							intra_X_num_edges++;
						}

						// Find degree sequence of X_final
						std::vector<degree_vertex_pair> d_added_edges_within_X;
						for(boost::tie(ui, ui_end) = vertices(X_sub); ui != ui_end; ++ui){
							//degree_vertex_pair degree_vertex( degree( *ui,X_final ), *ui ); // Anonymize induced X
							degree_vertex_pair degree_vertex( degree(X_sub.local_to_global(*ui), G), *ui ); // Anonymize X wrt G
							d_added_edges_within_X.push_back(degree_vertex);
						}
						sort(d_added_edges_within_X.begin(), d_added_edges_within_X.end(), compare_descending);
						//print_degree_sequence("Degree sequence of graph after added edges within X", d_added_edges_within_X);
						DEBUG_END("Adding edges from H to X, displaying final degree sequence ...");

						DEBUG_START("Find possible additional edges in G \\ X to make X k-anonymous...");
						// Find which vertices need more edges to become anonymized
						std::vector<degree_vertex_pair> unanonymized_vertices;
						size_t i = 0;
						for(dvit = d_added_edges_within_X.begin(); dvit < d_added_edges_within_X.end(); ++dvit){
							ptrdiff_t u_i = AnonymizedDegreeSequence.at(i++).first - (*dvit).first;
							if( u_i > 0 ){
								unanonymized_vertices.push_back( std::make_pair(u_i, (*dvit).second) );
							}
							else if( u_i < 0 ){
								log_file_verbose << "ERROR!!! Added too many edges to X such that the upper bounds of an edge exceeded the anonymized degree sequence." << std::endl;								
								exit_program();
							}
						}

						// If there are any unanonymized vertices, find possible edges to make X k-anonymous, otherwise exit noting success
						bool edges_found = true;
						
						size_t non_intra_X_num_edges = 0;
						std::vector<degree_vertex_pair> d_final;
						if( unanonymized_vertices.size() > 0 ){
							sort(unanonymized_vertices.begin(), unanonymized_vertices.end(), compare_descending);

							log_file_verbose << "Edges to add to X to make it k-degree anonymous (in X, in G \\ X):" << std::endl << std::endl;

							// Find list of all possible nodes in G which can be connected to an unanonymized vertex in X
							for(dvit = unanonymized_vertices.begin(); dvit < unanonymized_vertices.end(); ++dvit){
								size_t upper_bound = (*dvit).first;

								vertex_descriptor_t v = (*dvit).second;
								vertex_descriptor_t v_global = X_sub.local_to_global(v);
								std::vector<vertex_descriptor_t> possible_vertices;
								for(boost::tie(ui, ui_end) = vertices(G); ui != ui_end; ++ui){
									if( !edge(*ui, v_global, G).second && !X_sub.find_vertex(*ui).second ){	// Check that there's no edge from u to v and u is not in X_final
										possible_vertices.push_back(*ui);
									}
								}

								size_t num_of_added_edges = 0;
								while(num_of_added_edges < upper_bound){
									// Randomly add edges from G to X for unanonymized vertices in X
						
									double rand1 = uni01();
									if( possible_vertices.size() > 0 ){
										size_t random_index = (size_t)(rand1 * possible_vertices.size());
										log_file_verbose << "(" << v_global << "," << possible_vertices.at(random_index) << ")" << std::endl;
										add_edge(vertex(possible_vertices.at(random_index), G), v_global, G);
										possible_vertices.erase(possible_vertices.begin() + random_index);
										num_of_added_edges++;
										non_intra_X_num_edges++;
									}
									else{
										edges_found = false;
										break;
									}
								}
							}

							// Find degree sequence of X
							for(boost::tie(ui, ui_end) = vertices(X_sub); ui != ui_end; ++ui){
								degree_vertex_pair degree_vertex( degree(X_sub.local_to_global(*ui), G), *ui ); // Anonymize X wrt G
								d_final.push_back(degree_vertex);
							}
							sort(d_final.begin(), d_final.end(), compare_descending);

							//print_degree_sequence("Anonymized degree sequence", AnonymizedDegreeSequence);
							//print_degree_sequence("Degree sequence of final graph X", d_final);
						}
						else{

							d_final = d_added_edges_within_X;
							log_file_verbose << "No edges needed from G \\ X to X to make X k-anonymous." << std::endl;
						}

						// Graph is optimal if less than 2 edges added from G \\ X (could be optimal if 2 or more added, but it cannot be determined if it is)
						bool is_optimal = false;
						if( non_intra_X_num_edges <= 1 ){
							is_optimal = true;
							optimal_graphs++;
						}
						DEBUG_END("Find possible edges to add from G \\ X to X to make X k-anonymous...");

						DEBUG_START("Check if X is k-anonymous...");
						bool is_success;
						size_t number_failed_vertices = 0;
						size_t anonymization_failed_cost = 0;
						if( is_k_degree_anonymous(d_final, k) ){ // if( edges_found ){
							log_file_verbose << "SUCCESS: X made k-anonymous." << std::endl;
							successes++;
							is_success = true;
						}
						else{
							log_file_verbose << "FAIL: X not made k-anonymous." << std::endl;

							// Determine number of vertices failed and how much it would cost to anonymize them
							size_t i = 0;
							for(dvit = d_final.begin(); dvit < d_final.end(); ++dvit){
								ptrdiff_t u_i = AnonymizedDegreeSequence.at(i++).first - (*dvit).first;
								if( u_i > 0 ){
									number_failed_vertices++;
									anonymization_failed_cost += u_i;
								}
							}

							//DEBUG_START("Writing graph G to graph file in graphviz .dot format ...");
							//ss.str("");
							//ss << "graph_" << current_date_and_time << ".dot";
							//boost::iostreams::stream<boost::iostreams::file_sink> graph_file(ss.str());
							//write_graphviz(graph_file, G);
							//DEBUG_END("Writing graph G to graph file in graphviz .dot format ...");
							failures++;
							is_success = false;
						}
						DEBUG_END("Check if X is k-anonymous...");

						DEBUG_START("Check if G is k-anonymous...");
						std::vector<degree_vertex_pair> degree_sequence_G;
						for(boost::tie(ui, ui_end) = vertices(G); ui != ui_end; ++ui){
							degree_vertex_pair degree_vertex( degree(*ui, G), *ui ); 
							degree_sequence_G.push_back(degree_vertex);
						}
						sort(degree_sequence_G.begin(), degree_sequence_G.end(), compare_descending);

						if( is_k_degree_anonymous(degree_sequence_G, k) ){
							log_file_verbose << "G is k-anonymous." << std::endl;
						}
						else{
							log_file_verbose << "G is not k-anonymous." << std::endl;
						}
						DEBUG_END("Check if G is k-anonymous...");

						log_file_verbose << "Succeeded " << successes << " times and failed " << failures << " times." << std::endl;

						// Questions to answer for output to pertinent log
						//		1. Is optimal (i.e., UDCS satisfies all deficiencies)?
						//		2. Is a success (i.e., can complete anon with edges outside X)?
						//		3. Intra-X edges added?
						//		4. Edges added from X to V\X?
						//		5. # vertices not anonymized (assuming no to 2)?
						//		6. Sum deficiencies outstanding (again, assuming no to 2; >= 5)?
						//		7. Execution time (roughly, whether order 100ms, 1s, 10s, 100s, etc.)
						//		8. # augmenting paths found by Edmond's (if easy to measure, # matches final - # matches in greedy)? 
						//		9. Sum deficiencies after degree sequence anonymization?
						log_file_pertinent << "Is optimal: " << is_optimal << std::endl;
						log_file_pertinent << "Is a success: " << is_success << std::endl;
						log_file_pertinent << "Intra-X edges added: " << intra_X_num_edges << std::endl;	intra_X_num_edges_total += intra_X_num_edges;
						log_file_pertinent << "Edges added from X to G\\X: " << non_intra_X_num_edges << std::endl; non_intra_X_num_edges_total += non_intra_X_num_edges;
						if(!is_success){
							log_file_pertinent << "Number of vertices not anonymized: " << number_failed_vertices << std::endl; number_failed_vertices_total += number_failed_vertices;
							log_file_pertinent << "Sum of deficiencies outstanding: " << anonymization_failed_cost << std::endl; anonymization_failed_cost_total += anonymization_failed_cost;
						}
						double experiment_time_in_seconds = experiment_time.elapsed();
						log_file_pertinent << "Execution time: " << experiment_time_in_seconds << " seconds" << std::endl; experiment_time_in_seconds_total += experiment_time_in_seconds;
						//log_file_pertinent << "Number of augmenting paths found: " << std::endl; // Same as number 3, as only matching edges in substitute initially
						log_file_pertinent << "Sum of deficiencies after degree sequence anonymization: " << anonymization_cost << std::endl; anonymization_cost_total += anonymization_cost;
						log_file_pertinent << "Number of vertices in X_com: " << number_of_vertices_X_com << std::endl; number_of_vertices_X_com_total += number_of_vertices_X_com;
						log_file_pertinent << "Number of edges in X_com: " << number_of_edges_X_com << std::endl; number_of_edges_X_com_total += number_of_edges_X_com;
						log_file_pertinent << "Number of vertices in X_com': " << number_of_vertices_X_prime_actual << std::endl; number_of_vertices_X_prime_actual_total += number_of_vertices_X_prime_actual;
						log_file_pertinent << "Number of edges in X_com': " << number_of_edges_X_prime_actual << std::endl; number_of_edges_X_prime_actual_total += number_of_edges_X_prime_actual;
						log_file_pertinent << std::endl;
						//print_degree_sequence("Initial degree sequence", d, false);
						//print_degree_sequence("Final degree sequence", d_final, false);
						log_file_pertinent << "------------------------------------------------------------------------------" << std::endl;

						//ExpNo,|V|,k,|X|-pct,opt,success,Intra-ad,Extra-ad,time,sum_def,|V_com'|,|E_com'|
						experiment_data << current_experiment << "," 
										<< number_of_vertices << "," 
										<< k << "," 
										<< subset_X_percent << ","
										<< is_optimal << "," 
										<< is_success << "," 
										<< intra_X_num_edges << "," 
										<< non_intra_X_num_edges << ","
										<< experiment_time_in_seconds << "," 
										<< anonymization_cost << ","
										<< number_failed_vertices << ","
										<< anonymization_failed_cost << ","
										<< number_of_vertices_X_com << ","
										<< number_of_edges_X_com << ","
										<< number_of_vertices_X_prime_actual << ","
										<< number_of_edges_X_prime_actual
										<< std::endl;

						current_experiment++;
					}
					
					cout_and_log_file_pertinent << std::endl << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
					cout_and_log_file_pertinent << "End case:" << std::endl;
					cout_and_log_file_pertinent << "\tNumber of optimal graphs: " << optimal_graphs << std::endl;
					cout_and_log_file_pertinent << "\tNumber of successful graphs: " << successes << std::endl;
					cout_and_log_file_pertinent << "\tNumber of failed graphs: " << failures << std::endl;
					cout_and_log_file_pertinent << "\tAverage intra-X edges added: " << (double)intra_X_num_edges_total / number_of_experiments << std::endl;
					cout_and_log_file_pertinent << "\tEdges added from X to G\\X: " << (double)non_intra_X_num_edges_total / number_of_experiments << std::endl;
					cout_and_log_file_pertinent << "\tNumber of vertices not anonymized: " << (double)number_failed_vertices_total / number_of_experiments << std::endl;
					cout_and_log_file_pertinent << "\tSum of deficiencies outstanding: " << (double)anonymization_failed_cost_total / number_of_experiments << std::endl;
					cout_and_log_file_pertinent << "\tAverage execution time: " << (double)experiment_time_in_seconds_total / number_of_experiments << " seconds" << std::endl;
					cout_and_log_file_pertinent << "\tAverage sum of deficiencies after degree sequence anonymization: " << (double)anonymization_cost_total / number_of_experiments << std::endl;
					cout_and_log_file_pertinent << "\tNumber of vertices in X_com: " << (double)number_of_vertices_X_com_total / number_of_experiments << std::endl;
					cout_and_log_file_pertinent << "\tNumber of edges in X_com: " << (double)number_of_edges_X_com_total / number_of_experiments << std::endl;
					cout_and_log_file_pertinent << "\tNumber of vertices in X_com': " << (double)number_of_vertices_X_prime_actual_total / number_of_experiments << std::endl;
					cout_and_log_file_pertinent << "\tNumber of edges in X_com': " << (double)number_of_edges_X_prime_actual_total / number_of_experiments << std::endl;
					cout_and_log_file_pertinent << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;

					//Num Exp,|V|,k,|X|-pct,opt,success,Intra-ad,Extra-ad,time,sum_def,|V_com'|,|E_com'|
					experiment_data_averages << number_of_experiments << "," 
									<< number_of_vertices << "," 
									<< k << "," 
									<< subset_X_percent << ","
									<< optimal_graphs << "," 
									<< successes << "," 
									<< (double)intra_X_num_edges_total / number_of_experiments << ","
									<< (double)non_intra_X_num_edges_total / number_of_experiments << ","
									<< (double)experiment_time_in_seconds_total / number_of_experiments << "," 
									<< (double)anonymization_cost_total / number_of_experiments << ","
									<< (double)number_failed_vertices_total / number_of_experiments << ","
									<< (double)anonymization_failed_cost_total / number_of_experiments << ","
									<< (double)number_of_vertices_X_com_total / number_of_experiments << ","
									<< (double)number_of_edges_X_com_total / number_of_experiments << ","
									<< (double)number_of_vertices_X_prime_actual_total / number_of_experiments << ","
									<< (double)number_of_edges_X_prime_actual_total / number_of_experiments
									<< std::endl;
				}
			}
		}
	//} // Comment out when not using user input
}

std::vector< vertex_pair_t > upper_degree_constrained_subgraph(Graph G, std::vector<degree_vertex_pair> d, std::map<vertex_descriptor_t, ptrdiff_t> upper_bounds, std::map<vertex_descriptor_t, ptrdiff_t> delta, size_t number_of_vertices_G_prime, size_t number_of_edges_G_prime, size_t number_of_vertices_G_prime_actual, size_t number_of_edges_G_prime_actual){
	bool implicit_representation = false;	// Implicit representation doesn't actually create new vertices, just uses initial graph given
	Graph G_prime;
	if( implicit_representation ){
		DEBUG_START("Initializing implicit representation ...");
		copy_graph(G, G_prime);
		number_of_vertices_G_prime_actual = num_vertices(G_prime);
		number_of_edges_G_prime_actual = num_edges(G_prime);
	}
	else{
		DEBUG_START("Explicit representation: Substitute K_d,delta structure for each vertex in G', determine list of edges and initial matching ...");
	}
	std::vector<vertex_descriptor_t> initial_matching(number_of_vertices_G_prime_actual, boost::graph_traits<Graph>::null_vertex()); // Used if explicit representation as matching
	std::map<vertex_descriptor_t, ptrdiff_t> num_exposed_subvertices; // Number of exposed vertices if implicit representation
	
	log_file_verbose << "G' properties:" << std::endl;
	log_file_verbose << "\tNumber of vertices: " << number_of_vertices_G_prime << std::endl;
	log_file_verbose << "\tNumber of edges: " << number_of_edges_G_prime + num_edges(G) << std::endl; // Not quite right, as we erase edges adjacent to vertices with u_i == 0 in G before this
	log_file_verbose << "\tNumber of vertices (actually used): " << number_of_vertices_G_prime_actual << std::endl;
	log_file_verbose << "\tNumber of edges (actually used): " << number_of_edges_G_prime_actual << std::endl;	

	size_t total_anonymization_cost = 0;
	std::vector<vertex_descriptor_t> vertex_K_d_delta;	// Explicit representation: Keeps track of which vertex in G correspond to vertex indices in each K_d,delta
	std::set<vertex_descriptor_t> delta_zero_vertices;
	if( !implicit_representation ){
		// Explicit representation
		size_t index = 0, edge_index = 0;

		std::vector< std::pair<vertex_descriptor_t,vertex_descriptor_t> > edge_list_G_prime( number_of_edges_G_prime_actual - num_edges(G) );
		std::vector< std::pair< vertex_descriptor_t, std::pair<vertex_descriptor_t,vertex_descriptor_t> > > external_vertices_pairs; // Maps each new external vertex number to the pair (u,v) = e in G, where e is the edge for which the new vertex was made
		for(size_t i = 0; i < d.size(); i++){
			vertex_descriptor_t u = d.at(i).second;
			std::vector<vertex_descriptor_t> internal_vertices, external_vertices;
			adjacency_iterator_t ai, ai_end;
			 
			// If u_i <= 0 (== 0), then don't add that vertex and it's edges, as it cannot be in the matching anyways
			size_t current_upper_bound = upper_bounds[u];
			if( current_upper_bound > 0 ){
				for (boost::tie(ai, ai_end) = adjacent_vertices(u, G); ai != ai_end; ++ai){
					external_vertices.push_back(index);
					std::pair<vertex_descriptor_t,vertex_descriptor_t> external_pair;
					if( u < *ai ){
						external_pair = std::make_pair(u, *ai);
					}
					else{
						external_pair = std::make_pair(*ai, u);
					}
					external_vertices_pairs.push_back( std::make_pair( index, external_pair ) );
					index++;
					vertex_K_d_delta.push_back( u );
				}
				for(ptrdiff_t j = 0; j < delta[u]; j++){
					internal_vertices.push_back(index++);
					vertex_K_d_delta.push_back( u );
				}
				if( delta[u] <= 0 )
					delta_zero_vertices.insert(u);
				std::vector<vertex_descriptor_t>::iterator vit, vit2;
				for(vit = external_vertices.begin(); vit < external_vertices.end(); ++vit){
					for(vit2 = internal_vertices.begin(); vit2 < internal_vertices.end(); ++vit2){
						edge_list_G_prime[edge_index++] = std::make_pair(*vit, *vit2);

						// Set up initial matching
						if( initial_matching[*vit] == initial_matching[*vit2] ){ // Only way equality can hold is if initial_matching[*vit] == initial_matching[*vit2] == boost::graph_traits<Graph>::null_vertex()
							initial_matching[*vit] = *vit2;
							initial_matching[*vit2] = *vit;
						}
					}
				}
			}
		}
		DEBUG_END("Explicit representation: Substitute K_d,delta structure for each vertex in G', determine list of edges and initial matching ...");

		// Add all edges between the newly created vertices in G' that correspond to each vertex adjacent to another vertex in G
		DEBUG_START("Add edges between external vertices in G' ...");
		std::vector< std::pair<vertex_descriptor_t, std::pair<vertex_descriptor_t,vertex_descriptor_t> > >::iterator it, it2;
		for(it = external_vertices_pairs.begin(); it < external_vertices_pairs.end(); ++it){
			for(it2 = it+1; it2 < external_vertices_pairs.end(); it2++){
				if( (*it).second == (*it2).second ){
					edge_list_G_prime.push_back( std::make_pair( (*it).first, (*it2).first ) ); //add_edge( vertex((*it).first, G_prime), vertex((*it2).first, G_prime), G_prime);
					external_vertices_pairs.erase(it2);
					break;
				}
			}
		}
		DEBUG_END("Add edges between external vertices in G' ...");

		DEBUG_START("Creating graph G' ...");
		std::vector< std::pair<size_t,size_t> >::iterator eit;
		for(eit = edge_list_G_prime.begin(); eit < edge_list_G_prime.end(); ++eit){
			add_edge( (*eit).first, (*eit).second, G_prime );
		}

		if( !delta_zero_vertices.empty() ){			
			log_file_verbose << "Vertices in G with delta equal to zero: ";
			std::set< vertex_descriptor_t >::iterator sit;
			for(sit = delta_zero_vertices.begin(); sit != delta_zero_vertices.end(); ++sit){
				log_file_verbose << *sit << ",";
			}
			
			//print_degree_sequence("Corresponding vertices in K_d,delta", vertex_K_d_delta);

			//DEBUG_START("Writing graph G_prime to graph file in graphviz .dot format ...");
			//std::stringstream ss;
			//ss.str("");
			//ss << "G_prime.gv";
			//boost::iostreams::stream<boost::iostreams::file_sink> graph_file(ss.str());
			//write_graphviz(graph_file, G_prime, make_label_writer(&vertex_K_d_delta[0]));
			//DEBUG_END("Writing graph G_prime to graph file in graphviz .dot format ...");
			//exit_program();
		}
		DEBUG_END("Creating graph G' ...");
	}
	else{
		// Implicit representation
		vertex_iterator_t ui, ui_end;
		for(boost::tie(ui, ui_end) = vertices(G_prime); ui != ui_end; ++ui){
			// Number of exposed subvertices for each severed vertex is initially the upper bounds of each vertex
			num_exposed_subvertices[*ui] = upper_bounds[*ui];
			total_anonymization_cost += upper_bounds[*ui];
		}
		DEBUG_END("Initializing implicit representation ...");
	}
	
	std::vector< std::pair<vertex_descriptor_t,vertex_descriptor_t> > H;	// Contains degree-constrained subgraph
	DEBUG_START("Finding matching on G' ...");
	if( implicit_representation ){
		std::map<vertex_descriptor_t, std::set<vertex_descriptor_t> > mates;	// List of all possible mates in implicit representation of a vertex
		std::map<vertex_descriptor_t, std::set<vertex_descriptor_t> > mates_extra_greedy;	// List of all possible mates from greedy initial matching
		std::map<vertex_descriptor_t, std::set<vertex_descriptor_t> > mates_greedy;	// List of all possible mates from extra greedy initial matching
		
		DEBUG_START_L2("Implicit representation: Finding initial matching ...");
		std::map<vertex_descriptor_t, ptrdiff_t> num_exposed_subvertices_extra_greedy(num_exposed_subvertices);
		std::map<vertex_descriptor_t, ptrdiff_t> num_exposed_subvertices_greedy(num_exposed_subvertices);
		size_t initial_anonymized_cost_handled_extra_greedy = extra_greedy_implicit_initial_matching(G_prime, mates_extra_greedy, num_exposed_subvertices_extra_greedy);
		size_t initial_anonymized_cost_handled_greedy = greedy_implicit_initial_matching(G_prime, mates_greedy, num_exposed_subvertices_greedy);

		log_file_verbose << "\tTotal anonymization cost: " << total_anonymization_cost << std::endl;
		log_file_verbose << "\tTotal anonymization cost handled by initial matching (greedy): " << initial_anonymized_cost_handled_greedy << std::endl;
		log_file_verbose << "\tTotal anonymization cost handled by initial matching (extra greedy): " << initial_anonymized_cost_handled_extra_greedy << std::endl;

		size_t initial_anonymized_cost_handled = 0;
		if( initial_anonymized_cost_handled_extra_greedy > initial_anonymized_cost_handled_greedy ){
			mates = mates_extra_greedy;
			num_exposed_subvertices = num_exposed_subvertices_extra_greedy;
			initial_anonymized_cost_handled = initial_anonymized_cost_handled_extra_greedy;
			log_file_verbose << "\tUsing extra greedy initial matching." << std::endl;
		}
		else{
			mates = mates_greedy;
			num_exposed_subvertices = num_exposed_subvertices_greedy;
			initial_anonymized_cost_handled = initial_anonymized_cost_handled_greedy;
			log_file_verbose << "\tUsing greedy initial matching." << std::endl;
		}
		log_file_verbose << "\tCost remaining: " <<  total_anonymization_cost - initial_anonymized_cost_handled << std::endl;
		log_file_verbose << "\tMax number of augmenting paths to find: " <<  (total_anonymization_cost - initial_anonymized_cost_handled) / 2 << std::endl;
		print_degree_sequence("Number of exposed subvertices", num_exposed_subvertices);
		DEBUG_END_L2("Implicit representation: Finding initial matching ...");

		size_t augmenting_path_number = 0;	// Keeps track of number of times edmonds_implicit_matching is call (does not include recursive calls)
		DEBUG_START_L2("Find augmenting path ...");
		std::vector<vertex_descriptor_t> augmenting_path = edmonds_implicit_matching(G_prime, mates, num_exposed_subvertices);
		log_file_verbose << "\tAugmenting path number " << ++augmenting_path_number << " found ..." << std::endl;
		DEBUG_END_L2("Find augmenting path ...");

		while( !augmenting_path.empty() ){
			// Reverse matching on augmenting path
			bool change_to_matching = true;
			std::vector<vertex_descriptor_t>::iterator it;
			for(it = augmenting_path.begin(); it < augmenting_path.end()-1; ++it){
				if( change_to_matching ){
					mates[*(it+1)].insert(*it);
					mates[*it].insert(*(it+1));
					change_to_matching = false;
				}
				else{
					mates[*(it+1)].erase(*it);
					mates[*it].erase(*(it+1));
					change_to_matching = true;
				}
			}
			vertex_descriptor_t augmenting_path_start = augmenting_path.front();
			vertex_descriptor_t augmenting_path_end = augmenting_path.back();
			--num_exposed_subvertices[augmenting_path_start];
			--num_exposed_subvertices[augmenting_path_end];
			print_degree_sequence("Number of exposed subvertices", num_exposed_subvertices);

			DEBUG_START_L2("Find augmenting path ...");
			augmenting_path = edmonds_implicit_matching(G_prime, mates, num_exposed_subvertices);
			if( !augmenting_path.empty() ){
				log_file_verbose << "\tAugmenting path number " << ++augmenting_path_number << " found ..." << std::endl;
			}
			DEBUG_END_L2("Find augmenting path ...");
		}

		DEBUG_START_L2("Set edges to be added to G ...");
		total_anonymization_cost = 0;
		vertex_iterator_t ui, ui_end;
		for(boost::tie(ui,ui_end) = vertices(G_prime); ui != ui_end; ++ui){
			std::set<vertex_descriptor_t>::iterator sit;
			for(sit = mates[*ui].begin(); sit != mates[*ui].end(); ++sit){
				if(*ui < *sit){
					total_anonymization_cost += 2;
					H.push_back( std::make_pair(*ui, *sit) );
				}
			}
		}
		log_file_verbose << "\tAcutal cost of anonymizing handled in X only: " << total_anonymization_cost << std::endl;
		DEBUG_END_L2("Set edges to be added to G ...");
	}
	else{
		// Find maximum cardinality matching in G', which corresponds upper degree constained subgraph G
		//	http://www.boost.org/doc/libs/1_47_0/libs/graph/doc/maximum_matching.html

		if( num_edges(G_prime) > 0 ){
			//bool check_matching = matching<Graph, size_t *, property_std::map<Graph, vertex_index_t>::type,
			//	edmonds_augmenting_path_finder, input_initial_matching, maximum_cardinality_matching_verifier>(G_prime, &initial_matching[0], get(vertex_index,G_prime));
	//no_matching_verifier
			bool check_matching = boost_altered::checked_edmonds_maximum_cardinality_matching(G_prime, initial_matching, get(boost::vertex_index,G_prime));
			//boost_altered::edmonds_maximum_cardinality_matching(G_prime, initial_matching, get(vertex_index,G_prime));

			if( check_matching ){
				//log_file_verbose << "Maximum cardinality matching size: " << matching_size(G_prime, &initial_matching[0]) << std::endl;

				// Find edges to be added to G, which corresponds to the maximum cardinality matching in G'
				vertex_iterator_t ui, ui_end;
				for(boost::tie(ui,ui_end) = vertices(G_prime); ui != ui_end; ++ui){
					if(initial_matching[*ui] != boost::graph_traits<Graph>::null_vertex() && *ui < initial_matching[*ui]){
						size_t index_match_start = vertex_K_d_delta.at(*ui);	// Find which vertex in G u_i 
						size_t index_match_end = vertex_K_d_delta.at(initial_matching[*ui]);
						if(index_match_start != index_match_end){
							//log_file_verbose << "Add edge {" << index_match_start << ", " << index_match_end << "} to G to make it k-degree anonymous." << std::endl;
							H.push_back( std::make_pair(index_match_start, index_match_end) );
						}
					}
				}
			}
			//else{
			//	cout_and_log_file_pertinent << "ERROR!!! Maximum cardinality matching failed check." << std::endl;
			//	exit_program();
			//}
		}
	}
	DEBUG_END("Finding matching on G' ...");

	return H;
}

std::vector<vertex_descriptor_t> edmonds_implicit_matching(Graph G_prime, std::map<vertex_descriptor_t, std::set<vertex_descriptor_t> > mates, std::map<vertex_descriptor_t, ptrdiff_t> num_exposed_subvertices){

	std::vector<vertex_descriptor_t> augmenting_path;

	///////////////////////////////////////////////////////////////////////////////////
	//
	// Initial data structures to keep track of information for implicit matching
	//
	std::map< vertex_descriptor_t, std::vector<vertex_descriptor_t> > blossoms;
	std::vector<vertex_descriptor_t> parent( num_vertices(G_prime), boost::graph_traits<Graph>::null_vertex() );
	std::vector<vertex_descriptor_t> root( num_vertices(G_prime), boost::graph_traits<Graph>::null_vertex() );
	//std::vector< edge_descriptor_t > even_edges;
	std::vector<size_t> level( num_vertices(G_prime) );	// Level of vertex in F
	std::map<edge_descriptor_t, bool> marked_edges;
	std::map<vertex_descriptor_t, bool> marked_vertices;
	std::vector<vertex_descriptor_t> F;

	// Mark any matched edges
	edge_iterator_t ei, ei_end;
	for( boost::tie(ei, ei_end) = edges(G_prime); ei != ei_end; ++ei){
		vertex_descriptor_t v = source( *ei, G_prime );
		vertex_descriptor_t w = target( *ei, G_prime );
		if( mates[v].find(w) != mates[v].end() ){
			marked_edges[*ei] = true;
			edge_descriptor_t reverse_current_edge = edge( target(*ei, G_prime), source(*ei, G_prime), G_prime ).first;
			bool reverse_current_edge_exists = edge( target(*ei, G_prime), source(*ei, G_prime), G_prime ).second;
			if( reverse_current_edge_exists ){
				marked_edges[reverse_current_edge] = true;	// Make sure reverse ordering of edge is marked as true
			}
		}
		else{
			marked_edges[*ei] = false;
		}
	}

	// Add exposed vertices to F
	vertex_iterator_t vi, vi_end;
	size_t total_num_exposed_subvertices = 0;	// Keep track of the total number of exposed subvertices, if only 1, exit
	for(boost::tie(vi,vi_end) = vertices(G_prime); vi != vi_end; ++vi){
		vertex_descriptor_t v = *vi;
		if( num_exposed_subvertices[v] >= 1 ){
			total_num_exposed_subvertices += num_exposed_subvertices[v];
			// Set the parent, root and level of the the vertex v (where (v,w) is the current edge), where it is itself
			parent[v] = v;
			root[v] = v;
			level[v] = 0;

			// Set v as unmarked and add to F
			marked_vertices[v] = false;
			F.push_back(v);
		}
	}

	log_file_verbose << "\tedmonds_implicit_matching:  Total number of exposed subvertices: " << total_num_exposed_subvertices << std::endl;
	if( total_num_exposed_subvertices <= 1 ){
		return augmenting_path;
	}

	std::vector<vertex_descriptor_t>::iterator it;
	while( !F.empty() ){
		
		vertex_descriptor_t v = F.back();
		F.pop_back();

		// Check if vertex is marked or odd, skip
		if( marked_vertices[v] || level[v] % 2 == 1 ){
			continue;
		}

		out_edge_iterator_t ei, ei_end;
		for(boost::tie(ei,ei_end) = out_edges(v, G_prime); ei != ei_end; ++ei){
			// Current edge info
			edge_descriptor_t current_edge = *ei;
			vertex_descriptor_t w = target(current_edge, G_prime);

			// Check if edge is marked, skip
			if( marked_edges[current_edge] ){
				continue;
			}
			edge_descriptor_t reverse_current_edge = edge( target(*ei, G_prime), source(*ei, G_prime), G_prime ).first;
			bool reverse_current_edge_exists = edge( target(*ei, G_prime), source(*ei, G_prime), G_prime ).second;
			if( reverse_current_edge_exists && marked_edges[reverse_current_edge] ){
				continue;
			}


			///////////////////////////////////////////////////////////////////////////////////
			//
			// Cases of Edmond's algorithm (implicit), while there exists even unmarked edges
			//

			// Case 1: If w is not in F (if in F, root and parent are set)
			if( root[w] == boost::graph_traits<Graph>::null_vertex() ){
				// Mark the parent, root and level of the the vertex w
				parent[w] = v;
				root[w] = root[v];
				level[w] = level[v]+1;

				// Set w as unmarked and add to F
				marked_vertices[w] = false;
				F.push_back(w);

				std::set<vertex_descriptor_t> x_all = mates[w];	// As it is implicit representation, w can have 1 or more mates
				std::set<vertex_descriptor_t>::iterator sit;
				for(sit = x_all.begin(); sit != x_all.end(); ++sit){
					vertex_descriptor_t x = *sit;

					// x could already be in F, so check that it is not
					if( root[x] == boost::graph_traits<Graph>::null_vertex() ){ // True if x not in F
						// Mark the parent, root and level of the the vertex x (where (w,x) is the matched edge being added to F)
						parent[x] = w;
						root[x] = root[w];
						level[x] = level[w]+1;

						// Set x as unmarked and add to F
						marked_vertices[x] = false;
						F.push_back(x);
					}
					else{ // x is in F
						continue;
					}
				}
			}

			// Case 2: w is odd 
			else if( level[w] % 2 == 1 ){
				marked_edges[current_edge] = true;
				if( reverse_current_edge_exists){
					marked_edges[reverse_current_edge] = true;	// Make sure reverse ordering of edge is marked as true
				}
				continue;
			}

			// Case 3: Root of v is different that root of w, we have an augmenting path from exposed vertex to exposed vertex
			else if( root[v] != root[w] ){
				std::vector<vertex_descriptor_t> augmenting_path_v = edmonds_find_augmenting_path(v, root[v], parent); // Finds path from v to root[v] (must insert reverse of this into augmenting_path)							
				std::vector<vertex_descriptor_t> augmenting_path_w = edmonds_find_augmenting_path(w, root[w], parent); // Finds path from w to root[w]

				augmenting_path.insert( augmenting_path.end(), augmenting_path_v.rbegin(), augmenting_path_v.rend() );
				augmenting_path.insert( augmenting_path.end(), augmenting_path_w.begin(), augmenting_path_w.end() );

				print_augmenting_path("Augmenting path P (Case 3)", augmenting_path );
				return augmenting_path;
			}
			// Case 4: Found blossom
			// Case 4a: Blossom, but if w is root of blossom (root[v] == w) and has more than 2 exposed subvertices or 
			//				v is root of blossom (root[w] == v) and has more than 2 exposed subvertices, it's also an augmenting path, 
			//				so don't bother contracting (v or w must have more than two exposed subvertices, as it is matching augmentation adds two matching edges)
			else if( (root[v] == w && num_exposed_subvertices[w] >= 2) || (root[w] == v && num_exposed_subvertices[v] >= 2) ){
				std::vector<vertex_descriptor_t> augmenting_path_v = edmonds_find_augmenting_path(v, root[v], parent); // Finds path from v to root[v] (must insert reverse of this into augmenting_path)							
				std::vector<vertex_descriptor_t> augmenting_path_w = edmonds_find_augmenting_path(w, root[w], parent); // Finds path from w to root[w]

				augmenting_path.insert( augmenting_path.end(), augmenting_path_v.rbegin(), augmenting_path_v.rend() );
				augmenting_path.insert( augmenting_path.end(), augmenting_path_w.begin(), augmenting_path_w.end() );

				print_augmenting_path("Augmenting path P (Case 4a (w/v root of blossom))", augmenting_path );
				return augmenting_path;
			}
			// Case 4b: Blossom with common ancestor not equal to root[u]
			else{	
				// Find common ancestor of v and w, which is where augmenting_path_v and augmenting_path_w both have same vertex
				DEBUG_START_L2("Find blossom, contract and lift P' to P ... ");
				std::vector<vertex_descriptor_t> augmenting_path_v = edmonds_find_augmenting_path(v, root[v], parent); // Finds path from v to root[v]
				std::vector<vertex_descriptor_t> augmenting_path_w = edmonds_find_augmenting_path(w, root[w], parent); // Finds path from w to root[w]

				ptrdiff_t v_w_size_difference = augmenting_path_v.size() - augmenting_path_w.size();
				if( v_w_size_difference > 0 ){
					augmenting_path_v.erase( augmenting_path_v.begin(), augmenting_path_v.begin() + v_w_size_difference );
				}
				else if( v_w_size_difference < 0 ){
					augmenting_path_w.erase( augmenting_path_w.begin(), augmenting_path_w.begin() - v_w_size_difference );
				}

				while( augmenting_path_v.front() != augmenting_path_w.front() && !augmenting_path_v.empty() ){
					augmenting_path_v.erase(augmenting_path_v.begin());
					augmenting_path_w.erase(augmenting_path_w.begin());
				}

				if( augmenting_path_v.empty() || augmenting_path_v.empty() ){
					log_file_verbose << "ERROR! No common ancestor found for blossom." << std::endl;
					exit_program();
				}
				vertex_descriptor_t start_blossom = augmenting_path_v.front();
				//DEBUG_END_L2("Find start of blossom ... ");

				// Find blossom
				//DEBUG_START_L2("Find blossom ... ");
				std::vector<vertex_descriptor_t> blossom;	// Start vertex of blossom listed at beginning and end
				std::vector<vertex_descriptor_t> augmenting_path_v_start = edmonds_find_augmenting_path(v, start_blossom, parent); // Finds path from v to start_blossom
				std::vector<vertex_descriptor_t> augmenting_path_w_start = edmonds_find_augmenting_path(w, start_blossom, parent); // Finds path from w to start_blossom
				blossom.insert( blossom.end(), augmenting_path_v_start.rbegin(), augmenting_path_v_start.rend() );
				blossom.insert( blossom.end(), augmenting_path_w_start.begin(), augmenting_path_w_start.end() );
				print_augmenting_path("Blossom", blossom);
				//DEBUG_END_L2("Find blossom ... ");

				//DEBUG_START_L2("Contract blossom ... ");
				vertex_descriptor_t blossom_vertex = add_vertex(G_prime);
				log_file_verbose << "Contracted blossom vertex number: " << blossom_vertex << std::endl;

				std::vector<vertex_descriptor_t>::iterator it;
				std::set<vertex_descriptor_t> blossom_adjacent_vertices;
				for(it = blossom.begin(); it < blossom.end()-1; ++it){	// Don't check start_blossom twice (blossom.end()-1)
					vertex_descriptor_t blossom_u = *it;
					out_edge_iterator_t ei, ei_end;
					for(boost::tie(ei,ei_end) = out_edges(blossom_u, G_prime); ei != ei_end; ++ei){
						vertex_descriptor_t blossom_v = target(*ei, G_prime);

						// If blossom vertex was a mate to a vertex in G', set it to be a mate with the contracted blossom vertex
						if( mates[blossom_u].find(blossom_v) != mates[blossom_u].end() ){
							mates[blossom_v].insert(blossom_vertex);
							mates[blossom_vertex].insert(blossom_v);
						}
						blossom_adjacent_vertices.insert( blossom_v );	// Does not insert duplicates
					}
				}

				std::set<vertex_descriptor_t>::iterator sit;
				for(sit = blossom_adjacent_vertices.begin(); sit != blossom_adjacent_vertices.end(); ++sit){
					add_edge(blossom_vertex, *sit, G_prime);
				}

				// If the root (start_blossom) of the blossom was exposed (failed cased 4a due to having only 1 exposed subvertex), the blossom has 1 exposed subvertex, 0 otherwise
				num_exposed_subvertices[blossom_vertex] = num_exposed_subvertices[start_blossom];
				//root.push_back( root[start_blossom] );
				//parent.push_back( parent[start_blossom] );
				//level.push_back(0);

				// Remove all vertices in blossom (but remember for after exiting recursion), so they appear to not exist in matching algorithm
				std::vector<vertex_pair_t> prev_blossom_edges;
				for(it = blossom.begin(); it < blossom.end()-1; ++it){
					out_edge_iterator_t ei, ei_end;
					for(boost::tie(ei, ei_end) = out_edges(*it, G_prime); ei != ei_end; ++ei){
						vertex_descriptor_t u = source( *ei, G_prime );
						vertex_descriptor_t v = target( *ei, G_prime );
						if( u < v ){
							prev_blossom_edges.push_back( std::make_pair(u,v) );
						}
						else{
							prev_blossom_edges.push_back( std::make_pair(v,u) );
						}
					}
				}
				for(it = blossom.begin(); it < blossom.end()-1; ++it){
					clear_vertex(*it, G_prime);
				}

				//for(it = blossom.begin(); it < blossom.end()-1; ++it){
				//	clear_vertex(*it, G_prime);
				//	//mates[*it].clear();
				//	//num_exposed_subvertices[*it] = 0;
				//	//parent[*it] = boost::graph_traits<Graph>::null_vertex();
				//	//root[*it] = boost::graph_traits<Graph>::null_vertex();
				//	//level[*it] = 0;

				//	// Remove any edges in even_edges which have blossom vertices in them, replace with the contracted blossom vertex
				//	//replace(even_edges_source.begin(), even_edges_source.end(), *it, blossom_vertex);
				//	//replace(even_edges_target.begin(), even_edges_target.end(), *it, blossom_vertex);
				//}
				//DEBUG_END_L2("Contract blossom ... ");

				// Find P'
				DEBUG_RECURSION_START("Find P' (recursive call on edmonds_implicit_matching with contracted graph) ... ");
				augmenting_path = edmonds_implicit_matching(G_prime, mates, num_exposed_subvertices);
				print_augmenting_path("Augmenting path P'", augmenting_path);
				DEBUG_RECURSION_END("Find P' (recursive call on edmonds_implicit_matching with contracted graph) ... ");

				// Restore vertices/edges
				clear_vertex(blossom_vertex, G_prime);
				remove_vertex(blossom_vertex, G_prime);
				std::vector<vertex_pair_t>::iterator vvit;
				for(vvit = prev_blossom_edges.begin(); vvit < prev_blossom_edges.end(); ++vvit){
					add_edge( (*vvit).first, (*vvit).second, G_prime );
				}
				
				// Find P using P'
				//DEBUG_START_L2("Find P using P'");
				if( !augmenting_path.empty() ){
					std::vector<vertex_descriptor_t> augmenting_path_blossom_vertex_start;
					std::vector<vertex_descriptor_t> augmenting_path_in_blossom;
					std::vector<vertex_descriptor_t> augmenting_path_blossom_vertex_end;

					std::vector<vertex_descriptor_t>::iterator it;
					// Find path, not including blossom
					for(it = augmenting_path.begin(); it < augmenting_path.end(); ++it){
						if( *it == blossom_vertex ){
							augmenting_path_blossom_vertex_start.insert( augmenting_path_blossom_vertex_start.end(), augmenting_path.begin(), it );
							augmenting_path_blossom_vertex_end.insert( augmenting_path_blossom_vertex_end.end(), it+1, augmenting_path.end() );

							// Expand path through blossom, add to augmenting path
							vertex_descriptor_t into_blossom = boost::graph_traits<Graph>::null_vertex();
							vertex_descriptor_t exit_blossom = boost::graph_traits<Graph>::null_vertex();
							std::vector<vertex_descriptor_t>::iterator blossom_connector;
							std::vector<vertex_descriptor_t>::iterator it2;
							size_t blossom_position = 0;

							// Blossom at start of path
							if( it == augmenting_path.begin() ){
								exit_blossom = *(it+1);
								for(it2 = blossom.begin(); it2 < blossom.end()-1; ++it2){	// Don't check start_blossom twice (blossom.end()-1)
									if( edge(*it2, exit_blossom, G_prime).second ){
										blossom_connector = it2;
										break;
									}
									blossom_position++;
								}
							}
							// Blossom at end of path
							else if( it == augmenting_path.end() ){
								into_blossom = *(it-1);
								for(it2 = blossom.begin(); it2 < blossom.end()-1; ++it2){	// Don't check start_blossom twice (blossom.end()-1)
									if( edge(*it2, into_blossom, G_prime).second ){
										blossom_connector = it2;
										break;
									}
									blossom_position++;
								}
							}
							// Blossom in middle of path
							else{
								into_blossom = *(it-1);
								exit_blossom = *(it+1);
								for(it2 = blossom.begin(); it2 < blossom.end()-1; ++it2){	// Don't check start_blossom twice (blossom.end()-1)
									if( edge(*it2, exit_blossom, G_prime).second ){
										blossom_connector = it2;
										break;
									}
									blossom_position++;
								}
							}

							augmenting_path.clear();
							augmenting_path.insert( augmenting_path.end(), augmenting_path_blossom_vertex_start.begin(), augmenting_path_blossom_vertex_start.end() );
							if( blossom_position % 2 == 0 ){
								augmenting_path_in_blossom.insert( augmenting_path_in_blossom.end(), blossom.begin(), blossom_connector+1 );
								augmenting_path.insert( augmenting_path.end(), augmenting_path_in_blossom.begin(), augmenting_path_in_blossom.end() );
							}
							else{
								augmenting_path_in_blossom.insert( augmenting_path_in_blossom.end(), blossom_connector, blossom.end() );
								augmenting_path.insert( augmenting_path.end(), augmenting_path_in_blossom.rbegin(), augmenting_path_in_blossom.rend() );
							}
							augmenting_path.insert( augmenting_path.end(), augmenting_path_blossom_vertex_end.begin(), augmenting_path_blossom_vertex_end.end() );
							break;
						}
					}
				}
				
				print_augmenting_path("Augmenting path P", augmenting_path);
				//DEBUG_END_L2("Find P using P'");

				DEBUG_END_L2("Find blossom, contract and lift P' to P ... ");
				return augmenting_path;
			}

			marked_edges[current_edge] = true;
			if( reverse_current_edge_exists){
				marked_edges[reverse_current_edge] = true;	// Make sure reverse ordering of edge is marked as true
			}
		}

		marked_vertices[v] = false;
	}	

	// Didn't find path, so return empty path
	return augmenting_path;
}

size_t greedy_implicit_initial_matching(Graph G_prime, std::map<vertex_descriptor_t, std::set<vertex_descriptor_t> > &mates, std::map<vertex_descriptor_t, ptrdiff_t> &num_exposed_subvertices){
	edge_iterator_t ei, ei_end;
	size_t initial_anonymized_cost_handled = 0;	// Keep track of the anonymized cost handled by initial matching
	for( boost::tie(ei, ei_end) = edges(G_prime); ei != ei_end; ++ei){
		edge_descriptor_t e = *ei;
		vertex_descriptor_t u = source(e, G_prime);
		vertex_descriptor_t v = target(e, G_prime);

		if(num_exposed_subvertices[u] >= 1 && num_exposed_subvertices[v] >= 1){
			std::pair< std::set<vertex_descriptor_t>::iterator, bool > ret;

			ret = mates[u].insert(v);
			if(ret.second == true){	// True if inserted, false if it already exists
				initial_anonymized_cost_handled++;
				num_exposed_subvertices[u]--;
			}

			ret = mates[v].insert(u);
			if(ret.second == true){	// True if inserted, false if it already exists
				initial_anonymized_cost_handled++;
				num_exposed_subvertices[v]--;
			}
		}
	}
	return initial_anonymized_cost_handled;
}
// From Boost:
//  The "extra greedy matching" is formed by repeating the following procedure as many times as possible: Choose the
//  unmatched vertex v of minimum non-zero degree.  Choose the neighbor w of v which is unmatched 
//  and has minimum degree over all of v's neighbors. Add (u,v) to the matching. boost::ties for either 
//  choice are broken arbitrarily. This procedure takes time O(m log n), where m is the number of edges in the graph 
//  and n is the number of vertices.

// Helper functions
struct select_first
{
	inline static vertex_descriptor_t select_vertex(const vertex_pair_t p){
		return p.first;
	}
};

struct select_second
{
	inline static vertex_descriptor_t select_vertex(const vertex_pair_t p){
		return p.second;
	}
};

template <class PairSelector>
class less_than_by_degree
{
public:
	less_than_by_degree(const Graph& g): m_g(g) {}
	bool operator() (const vertex_pair_t x, const vertex_pair_t y)
	{
		return 
			out_degree(PairSelector::select_vertex(x), m_g) 
			< out_degree(PairSelector::select_vertex(y), m_g);
	}
private:
	const Graph& m_g;
};

size_t extra_greedy_implicit_initial_matching(Graph G_prime, std::map<vertex_descriptor_t, std::set<vertex_descriptor_t> > &mates, std::map<vertex_descriptor_t, ptrdiff_t> &num_exposed_subvertices){
	std::vector< vertex_pair_t > edge_list;

	edge_iterator_t ei, ei_end;
	for( boost::tie(ei, ei_end) = edges(G_prime); ei != ei_end; ++ei){
		edge_descriptor_t e = *ei;
		vertex_descriptor_t u = source(e, G_prime);
		vertex_descriptor_t v = target(e, G_prime);

		edge_list.push_back( std::make_pair(u,v) );
		edge_list.push_back( std::make_pair(v,u) );
	}

	// Sort the edges by the degree of the target, then (using a stable sort) by degree of the source
	sort(edge_list.begin(), edge_list.end(), less_than_by_degree<select_second>(G_prime) );
	stable_sort(edge_list.begin(), edge_list.end(), less_than_by_degree<select_first>(G_prime) );
      
	// Construct the extra greedy matching
	size_t initial_anonymized_cost_handled = 0;	// Keep track of the anonymized cost handled by initial matching
	for(std::vector< vertex_pair_t >::const_iterator it = edge_list.begin(); it != edge_list.end(); ++it)
	{
		if( num_exposed_subvertices[it->first] >= 1 && num_exposed_subvertices[it->second] >= 1 ){
			std::pair< std::set<vertex_descriptor_t>::iterator, bool > ret;

			ret = mates[it->first].insert(it->second);
			if(ret.second == true){	// True if inserted, false if it already exists
				initial_anonymized_cost_handled++;
				num_exposed_subvertices[it->first]--;
			}
			
			ret = mates[it->second].insert(it->first);
			if(ret.second == true){	// True if inserted, false if it already exists
				initial_anonymized_cost_handled++;
				num_exposed_subvertices[it->second]--;
			}
		}
	}
	return initial_anonymized_cost_handled;
}

// Find path from u to v
std::vector<vertex_descriptor_t> edmonds_find_augmenting_path(vertex_descriptor_t u, vertex_descriptor_t v, std::vector<vertex_descriptor_t> parent){
	std::vector<vertex_descriptor_t> augmenting_path_u;
	augmenting_path_u.push_back(u);
	if( u != v ){
		vertex_descriptor_t current_vertex = u;
		vertex_descriptor_t up_vertex = parent[current_vertex];
		while( up_vertex != v ){
			augmenting_path_u.push_back(up_vertex);
			up_vertex = parent[up_vertex];
		}
		augmenting_path_u.push_back(v);
	}

	return augmenting_path_u;
}

void DEBUG_RECURSION_START(std::string debug_message){
	log_file_verbose << std::endl << "\tStart (recursion level " << recursion_level++ << "): " << debug_message << std::endl << std::endl;
}

void DEBUG_RECURSION_END(std::string debug_message){
	log_file_verbose << std::endl << "\tEnd (recursion level " << --recursion_level << "): " << debug_message << std::endl << std::endl;
}

void DEBUG_START(std::string debug_message){
	t.restart();
	log_file_verbose << std::endl << "Start: " << debug_message << std::endl << std::endl;
}

void DEBUG_END(std::string debug_message){
	log_file_verbose << std::endl << "End: " << debug_message << " (Took " << t.elapsed() << " seconds)" << std::endl << std::endl;
}

void DEBUG_START_L2(std::string debug_message){
	t_l2.restart();
	log_file_verbose << std::endl << "\tStart: " << debug_message << std::endl << std::endl;
}

void DEBUG_END_L2(std::string debug_message){
	log_file_verbose << std::endl << "\tEnd: " << debug_message << " (Took " << t_l2.elapsed() << " seconds)" << std::endl << std::endl;
}

void exit_program(){
	std::cout << "Press ENTER to exit.";
	std::cout.flush();
	std::string input = "";
	std::getline(std::cin, input);
	exit(1);
}

void select_input_graph(size_t& input_graph){
	///////////////////////////////////////////////////////////////////////////////////
	//
	// Get input graph choice
	//
	std::string input = "";
	while(1){
		std::cout << std::endl << "Enter input graph, " << SMALL_WORLD_GRAPH << " for " << graph_titles[SMALL_WORLD_GRAPH] 
			<< ", " << PLOD_GRAPH << " for " << graph_titles[PLOD_GRAPH]
			<< ", " << ENRON_GRAPH << " for " << graph_titles[ENRON_GRAPH]
			<< ", " << KARATE_GRAPH << " for " << graph_titles[KARATE_GRAPH]
			<< ", " << POWERGRID_GRAPH << " for " << graph_titles[POWERGRID_GRAPH]
			<< ", " << WIKI_GRAPH << " for " << graph_titles[WIKI_GRAPH]
			<< ", " << EPINIONS_GRAPH << " for " << graph_titles[EPINIONS_GRAPH]
			<< "(0 to EXIT, ENTER for default<" << graph_titles[default_input_graph] << ">): ";
			std::cout.flush();
		std::getline(std::cin, input);

		if( input.empty() ){
			input_graph = default_input_graph;
			break;
		}

		// This code converts from std::string to number safely.
		std::stringstream myStream(input);

		if (myStream >> input_graph){
			if( input_graph == 0 ){
				exit_program();
			}
			else if( input_graph != SMALL_WORLD_GRAPH && input_graph != PLOD_GRAPH && input_graph != ENRON_GRAPH && input_graph != KARATE_GRAPH && input_graph != POWERGRID_GRAPH && input_graph != WIKI_GRAPH && input_graph != EPINIONS_GRAPH ){
				std::cout << "Invalid number, must be either " << SMALL_WORLD_GRAPH << ", " << PLOD_GRAPH << ", " << ENRON_GRAPH << ", " << KARATE_GRAPH << ", " << POWERGRID_GRAPH << ", " << WIKI_GRAPH << ", or" << EPINIONS_GRAPH << std::endl;
				continue;
			}
			else{
				default_input_graph = input_graph;
				break;
			}
		}
		std::cout << "Invalid number, must be either " << SMALL_WORLD_GRAPH << ", " << PLOD_GRAPH << ", " << ENRON_GRAPH << ", " << KARATE_GRAPH << ", " << POWERGRID_GRAPH << ", " << WIKI_GRAPH << ", or" << EPINIONS_GRAPH << std::endl;
	}
	std::cout << "You chose: " << graph_titles[input_graph] << std::endl << std::endl;
}

template<class input_type>
void get_input_list(const std::string& description, std::vector<input_type>& input_list){
	std::string cin_input = "";
	std::cout << "Enter list of values for " << description << " (separate values by comma or tab): ";
	std::cout.flush();
	getline(std::cin, cin_input);

	if( cin_input.empty() ){
		std::cout << "ERROR!!! Nothing entered into list." << std::endl;
		exit_program();
	}

	std::vector<std::string> tokens;
	boost::split(tokens, cin_input, boost::is_any_of(",\t"));

	for(std::vector<std::string>::iterator it = tokens.begin(); it != tokens.end(); ++it){
		try{
			input_list.push_back(boost::lexical_cast<input_type>(*it));
		}
		catch(boost::bad_lexical_cast&){
			std::cout << "ERROR!!! Boost lexical_cast failed." << std::endl;
			exit_program();
		}
	}
	std::cout << "You entered: " << cin_input << std::endl << std::endl;
}

template<class input_type>
void get_input(const std::string& description, input_type& input){
	std::string cin_input = "";
	std::cout << "Enter values for " << description << ": ";
	std::cout.flush();
	getline(std::cin, cin_input);

	if( cin_input.empty() ){
		std::cout << "ERROR!!! Nothing entered." << std::endl;
		exit_program();
	}

	std::vector<std::string> tokens;
	boost::split(tokens, cin_input, boost::is_any_of(",\t"));

	for(std::vector<std::string>::iterator it = tokens.begin(); it != tokens.end(); ++it){
		try{
			input = boost::lexical_cast<input_type>(*it);
		}
		catch(boost::bad_lexical_cast&){
			std::cout << "ERROR!!! Boost lexical_cast failed." << std::endl;
			exit_program();
		}
	}
	std::cout << "You entered: " << cin_input << " for " << description << "." << std::endl << std::endl;
}

//void get_inputs(size_t& k, double& subset_X_percent){
//
//	///////////////////////////////////////////////////////////////////////////////////
//	//
//	// Get k value, check if valid
//	//
//	std::string input = "";
//	while(1){
//		cout_and_log_file_pertinent << "Enter value for k (0 to EXIT, ENTER for default<" << default_k << ">): ";
//		cout_and_log_file_pertinent.flush();
//		getline(std::cin, input);
//
//		if( input.empty() ){
//			k = default_k;
//			break;
//		}
//
//		// This code converts from std::string to number safely.
//		std::stringstream myStream(input);
//
//		if (myStream >> k){
//			if( k <= 1 ){
//				if( k == 0 ){
//					exit_program();
//				}
//				else{
//					cout_and_log_file_pertinent << "Invalid number, must be greater than 1" << std::endl;
//					continue;
//				}
//			}
//			else{
//				default_k = k;
//				break;
//			}
//		}
//		cout_and_log_file_pertinent << "Invalid number, please try again" << std::endl;
//	}
//	cout_and_log_file_pertinent << "You entered: " << k << std::endl << std::endl;
//
//	///////////////////////////////////////////////////////////////////////////////////
//	//
//	// Get subset_X_percent value, check if valid
//	//
//	input = "";
//	while(1){
//		cout_and_log_file_pertinent << "Enter value for subset X percent of G (0 to EXIT, ENTER for default<" << default_subset_X_percent << ">): ";
//		cout_and_log_file_pertinent.flush();
//		getline(std::cin, input);
//
//		if( input.empty() ){
//			subset_X_percent = default_subset_X_percent;
//			break;
//		}
//
//		// This code converts from std::string to number safely.
//		std::stringstream myStream(input);
//
//		if (myStream >> subset_X_percent){
//			if( subset_X_percent <= 0.0){
//				exit_program();
//			}
//			else if( subset_X_percent > 1.0 ){
//				cout_and_log_file_pertinent << "Invalid number, must be greater than 0 and less than or equal to 1.0" << std::endl;
//				continue;
//			}
//			else{
//				default_subset_X_percent = subset_X_percent;
//				break;
//			}
//		}
//		cout_and_log_file_pertinent << "Invalid number, please try again" << std::endl;
//	}
//	cout_and_log_file_pertinent << "You entered: " << subset_X_percent << std::endl << std::endl;
//}
//
//void get_inputs(double& alpha, double& beta){
//	///////////////////////////////////////////////////////////////////////////////////
//	//
//	// Get alpha
//	//
//	std::string input = "";
//	while(1){
//		cout_and_log_file_pertinent << std::endl << "Alpha for scale-free graph (0 to EXIT, ENTER for default<" << default_alpha << ">): ";
//		cout_and_log_file_pertinent.flush();
//		getline(std::cin, input);
//
//		if( input.empty() ){
//			alpha = default_alpha;
//			break;
//		}
//
//		// This code converts from std::string to number safely.
//		std::stringstream myStream(input);
//
//		if (myStream >> alpha){
//			if( alpha <= 0.0 ){
//				exit_program();
//			}
//			else if( alpha > 3.0 || alpha <= 2.0 ){
//				cout_and_log_file_pertinent << "Invalid number, must be greater than 1 and less than or equal to 3.0 (scale-free generally between 2 and 3)" << std::endl;
//				continue;
//			}
//			else{
//				default_alpha = alpha;
//				break;
//			}
//		}
//		cout_and_log_file_pertinent << "Invalid number, please try again" << std::endl;
//	}
//	cout_and_log_file_pertinent << "You entered: " << alpha << std::endl << std::endl;
//
//	///////////////////////////////////////////////////////////////////////////////////
//	//
//	// Get beta
//	//
//	input = "";
//	while(1){
//		cout_and_log_file_pertinent << std::endl << "Beta for scale-free graph (0 to EXIT, ENTER for default<" << default_beta << ">): ";
//		cout_and_log_file_pertinent.flush();
//		getline(std::cin, input);
//
//		if( input.empty() ){
//			beta = default_beta;
//			break;
//		}
//
//		// This code converts from std::string to number safely.
//		std::stringstream myStream(input);
//
//		if (myStream >> beta){
//			if( beta <= 0.0 ){
//				exit_program();
//			}
//			else if(beta <= 1.0){
//				cout_and_log_file_pertinent << "Invalid number, must be greater than 1.0" << std::endl;
//				continue;
//			}
//			else{
//				default_beta = beta;
//				break;
//			}
//		}
//		cout_and_log_file_pertinent << "Invalid number, please try again" << std::endl;
//	}
//	cout_and_log_file_pertinent << "You entered: " << beta << std::endl << std::endl;
//}
//
//void get_inputs(size_t& number_of_experiments, size_t& k, double& subset_X_percent, size_t& number_of_vertices, size_t& k_nearest_neighbors){
//	///////////////////////////////////////////////////////////////////////////////////
//	//
//	// Get number of experiments to run, check if valid
//	//
//	std::string input = "";
//	while(1){
//		cout_and_log_file_pertinent << std::endl << "Enter number of experiments to run (0 to EXIT, ENTER for default<" << default_number_of_experiments << ">): ";
//		cout_and_log_file_pertinent.flush();
//		getline(std::cin, input);
//
//		if( input.empty() ){
//			number_of_experiments = default_number_of_experiments;
//			break;
//		}
//
//		// This code converts from std::string to number safely.
//		std::stringstream myStream(input);
//
//		if (myStream >> number_of_experiments){
//			if( number_of_experiments <= 0 ){
//				if( number_of_experiments == 0 ){
//					exit_program();
//				}
//				else{
//					cout_and_log_file_pertinent << "Invalid number, must be greater than 0" << std::endl;
//					continue;
//				}
//			}
//			else{
//				default_number_of_experiments = number_of_experiments;
//				break;
//			}
//		}
//		cout_and_log_file_pertinent << "Invalid number, please try again" << std::endl;
//	}
//	cout_and_log_file_pertinent << "You entered: " << number_of_experiments << std::endl << std::endl;
//
//	// get k and subset_X_percent from user
//	get_inputs(k, subset_X_percent);
//
//	///////////////////////////////////////////////////////////////////////////////////
//	//
//	// Get number of vertices, check if valid
//	//
//	input = "";
//	while(1){
//		cout_and_log_file_pertinent << std::endl << "Enter number of vertices (0 to EXIT, ENTER for default<" << default_number_of_vertices << ">): ";
//		cout_and_log_file_pertinent.flush();
//		getline(std::cin, input);
//
//		if( input.empty() ){
//			number_of_vertices = default_number_of_vertices;
//			break;
//		}
//
//		// This code converts from std::string to number safely.
//		std::stringstream myStream(input);
//
//		if (myStream >> number_of_vertices){
//			if( number_of_vertices <= 1 ){
//				if( number_of_vertices == 0 ){
//					exit_program();
//				}
//				else{
//					cout_and_log_file_pertinent << "Invalid number, must be greater than 1" << std::endl;
//					continue;
//				}
//			}
//			else{
//				default_number_of_vertices = number_of_vertices;
//				break;
//			}
//		}
//		cout_and_log_file_pertinent << "Invalid number, please try again" << std::endl;
//	}
//	cout_and_log_file_pertinent << "You entered: " << number_of_vertices << std::endl << std::endl;
//
//	///////////////////////////////////////////////////////////////////////////////////
//	//
//	// Get k_nearest_neighbors_percent value, check if valid
//	//
//	input = "";
//	while(1){
//		cout_and_log_file_pertinent << "Each vertex connected to its k-nearest neighbors in small-world graph.  Enter value for k-nearest neighbors (0 to EXIT, ENTER for default<" << default_k_nearest_neighbors << ">): ";
//		cout_and_log_file_pertinent.flush();
//		getline(std::cin, input);
//
//		if( input.empty() ){
//			k_nearest_neighbors = default_k_nearest_neighbors;
//			break;
//		}
//
//		// This code converts from std::string to number safely.
//		std::stringstream myStream(input);
//
//		if (myStream >> k_nearest_neighbors){
//			if( k_nearest_neighbors < 1 || k_nearest_neighbors >= number_of_vertices ){
//				if( k_nearest_neighbors == 0 ){
//					exit_program();
//				}
//				else{
//					cout_and_log_file_pertinent << "Invalid number, must be between 1 and " << number_of_vertices - 1 << std::endl;
//					continue;
//				}
//			}
//			else{
//				default_k_nearest_neighbors = k_nearest_neighbors;
//				break;
//			}
//		}
//		cout_and_log_file_pertinent << "Invalid number, please try again" << std::endl;
//	}
//	cout_and_log_file_pertinent << "You entered: " << k_nearest_neighbors << std::endl << std::endl;
//	cout_and_log_file_pertinent.flush();
//}
//
//void get_inputs(size_t& number_of_experiments, size_t& k, double& subset_X_percent, size_t& number_of_vertices, double& alpha, double& beta){
//	///////////////////////////////////////////////////////////////////////////////////
//	//
//	// Get number of experiments to run, check if valid
//	//
//	std::string input = "";
//	while(1){
//		cout_and_log_file_pertinent << std::endl << "Enter number of experiments to run (0 to EXIT, ENTER for default<" << default_number_of_experiments << ">): ";
//		cout_and_log_file_pertinent.flush();
//		getline(std::cin, input);
//
//		if( input.empty() ){
//			number_of_experiments = default_number_of_experiments;
//			break;
//		}
//
//		// This code converts from std::string to number safely.
//		std::stringstream myStream(input);
//
//		if (myStream >> number_of_experiments){
//			if( number_of_experiments <= 0 ){
//				if( number_of_experiments == 0 ){
//					exit_program();
//				}
//				else{
//					cout_and_log_file_pertinent << "Invalid number, must be greater than 0" << std::endl;
//					continue;
//				}
//			}
//			else{
//				default_number_of_experiments = number_of_experiments;
//				break;
//			}
//		}
//		cout_and_log_file_pertinent << "Invalid number, please try again" << std::endl;
//	}
//	cout_and_log_file_pertinent << "You entered: " << number_of_experiments << std::endl << std::endl;
//
//	// get k and subset_X_percent from user
//	get_inputs(k, subset_X_percent);
//
//	///////////////////////////////////////////////////////////////////////////////////
//	//
//	// Get number of vertices, check if valid
//	//
//	input = "";
//	while(1){
//		cout_and_log_file_pertinent << std::endl << "Enter number of vertices (0 to EXIT, ENTER for default<" << default_number_of_vertices << ">): ";
//		cout_and_log_file_pertinent.flush();
//		getline(std::cin, input);
//
//		if( input.empty() ){
//			number_of_vertices = default_number_of_vertices;
//			break;
//		}
//
//		// This code converts from std::string to number safely.
//		std::stringstream myStream(input);
//
//		if (myStream >> number_of_vertices){
//			if( number_of_vertices <= 1 ){
//				if( number_of_vertices == 0 ){
//					exit_program();
//				}
//				else{
//					cout_and_log_file_pertinent << "Invalid number, must be greater than 1" << std::endl;
//					continue;
//				}
//			}
//			else{
//				default_number_of_vertices = number_of_vertices;
//				break;
//			}
//		}
//		cout_and_log_file_pertinent << "Invalid number, please try again" << std::endl;
//	}
//	cout_and_log_file_pertinent << "You entered: " << number_of_vertices << std::endl << std::endl;
//
//	///////////////////////////////////////////////////////////////////////////////////
//	//
//	// Get alpha and beta
//	//
//	get_inputs(alpha, beta);
//}

// Print augmenting path (or blossom)
void print_augmenting_path(const std::string& description, std::vector<vertex_descriptor_t> augmenting_path){
	log_file_verbose << std::endl;
	log_file_verbose << description << ": " << std::endl;
	if( !augmenting_path.empty() ){
		std::vector<vertex_descriptor_t>::iterator it;
		for(it = augmenting_path.begin(); it < augmenting_path.end()-1; ++it){
			log_file_verbose << *it << " -> ";
		}
		log_file_verbose << *it << std::endl;
	}
	else{
		log_file_verbose << "Augmenting path is empty." << std::endl;
	}
}

void print_degree_sequence(const std::string& description, std::vector<size_t> d){
	log_file_verbose << std::endl;
	log_file_verbose << description << ": " << std::endl;
	std::vector<size_t>::iterator it;
	for(it = d.begin(); it < d.end()-1; ++it){
		log_file_verbose << *it << "\t";
	}
	log_file_verbose << *it << std::endl;
}

void print_degree_sequence(const std::string& description, std::vector<degree_vertex_pair> d, bool verbose){
	if(verbose){
		log_file_verbose << std::endl;
		log_file_verbose << description << " (vertex id below in brackets): " << std::endl;
		std::vector<degree_vertex_pair>::iterator it;
		for(it = d.begin(); it < d.end()-1; ++it){
			log_file_verbose << (*it).first << "\t";
		}
		log_file_verbose << (*it).first << std::endl;
		for(it = d.begin(); it < d.end()-1; ++it){
			log_file_verbose << "(" << (*it).second << ")" << "\t";
		}
		log_file_verbose << "(" << (*it).second << ")" << std::endl;
	}
	else{
		log_file_pertinent << description << " (number of repetitions in brackets): ";
		std::vector<degree_vertex_pair>::iterator it;
		size_t current_degree = d.front().first;
		size_t k_grouping_size = 1;
		for(it = d.begin()+1; it < d.end()-1; ++it){
			if( current_degree != (*it).first ){
				log_file_pertinent << current_degree << "(" << k_grouping_size << "),";
				current_degree = (*it).first;
				k_grouping_size = 0;
			}
			k_grouping_size++;
		}

		if( current_degree != (*it).first ){
			log_file_pertinent << current_degree << "(" << k_grouping_size << "),";
			log_file_pertinent << (*it).first << "(1)" << std::endl;
		}
		else{
			log_file_pertinent << current_degree << "(" << ++k_grouping_size << ")" << std::endl;
		}
	}
}

void print_degree_sequence(const std::string& description, std::map<vertex_descriptor_t, ptrdiff_t> d){
	log_file_verbose << std::endl;
	log_file_verbose << description << " (vertex id below in brackets): " << std::endl;
	std::map<vertex_descriptor_t, ptrdiff_t>::iterator mit;
	std::vector<vertex_descriptor_t> vertex_numbers;
	for(mit = d.begin(); mit != d.end(); ++mit){
		mit++;
		if( mit == d.end() ){
			mit--;
			if( (*mit).second != 0 ){
				log_file_verbose << (*mit).second << std::endl;
				vertex_numbers.push_back( (*mit).first );
			}
		}
		else{
			mit--;
			if( (*mit).second != 0 ){
				log_file_verbose << (*mit).second << "\t";
				vertex_numbers.push_back( (*mit).first );
			}
		}
		
	}
	log_file_verbose << std::endl;
	if( !vertex_numbers.empty() ){
		std::vector<size_t>::iterator vit;
		for(vit = vertex_numbers.begin(); vit < vertex_numbers.end()-1; ++vit){
			log_file_verbose << "(" << *vit << ")" << "\t";
		}
		log_file_verbose << "(" << *vit << ")" << std::endl;
	}
	else{
		log_file_verbose << "All values are 0." << std::endl;
	}
}

bool is_k_degree_anonymous(std::vector<degree_vertex_pair> degree_sequence, size_t k){
	std::vector<degree_vertex_pair>::iterator it = degree_sequence.begin();
	size_t current_degree = (*it).first;
	size_t k_grouping_size = 1;
	for(; it < degree_sequence.end(); ++it){
		if( current_degree == (*it).first ){
			k_grouping_size++;
		}
		else{
			if( k_grouping_size >= k ){
				current_degree = (*it).first;
				k_grouping_size = 1;
			}
			else{
				return false;
			}
		}
	}
	return true;
}

///////////////////////////////////////////////////////////////////////////////////
//
// Check if the degree sequence represents a real graph (Erdos-Gallei Theorem)
//	for all 1 <= j <= n: sum{i=1 to j} d_i <= j(j-1) sum{i=j+1 to n} min(d_i,j)
//
size_t ErdosGallaiThm(std::vector< std::vector<degree_vertex_pair> > SetOfDegreeSequences, size_t n)
{
	std::vector<degree_vertex_pair>::iterator it;
	std::vector< std::vector<degree_vertex_pair> >::iterator it2;
	for(it2 = SetOfDegreeSequences.begin(); it2 < SetOfDegreeSequences.end(); it2++ ){
		for(size_t j = 1; j <= n; j++){
			size_t sum = 0;
			for(size_t i = 1; i <= j; i++){
				sum += (*it2).at(i-1).first;
			}

			size_t sum_2 = j*(j-1);
			for(size_t i = j+1; i <= n; i++){
				sum_2 += std::min((*it2).at(i-1).first,j);
			}

			if( sum > sum_2 ){
				log_file_verbose << "Degree sequence does not represent a real graph, fails at d_" << j << ": ";
				for(it = (*it2).begin(); it < (*it2).end()-1; ++it){
					log_file_verbose << (*it).first << ",";
				}
				log_file_verbose << (*it).first << std::endl;
				log_file_verbose << "Press ENTER to continue... " << std::flush;
				std::cin.ignore( std::numeric_limits<std::streamsize>::max(), '\n' );
				log_file_verbose << std::endl << std::endl;
				return 0;
			}

			if( j == n && sum % 2 != 0 ){
				log_file_verbose << "Degree sequence does not represent a real graph, sum of degrees is not even." << std::endl;
				log_file_verbose << "Press ENTER to continue... " << std::flush;
				std::cin.ignore( std::numeric_limits<std::streamsize>::max(), '\n' );
				log_file_verbose << std::endl << std::endl;
				return 0;
			}
		}
	}
	log_file_verbose << std::endl << std::endl;
	return 1;
}

///////////////////////////////////////////////////////////////////////////////////
//
//	Determine the total number of possible k-groupings
//		Isn't quite correct, as should check for when degree values are the same and can be size_terchanged
//
size_t NumberOfKGroupings(size_t total, std::vector<degree_vertex_pair> d_reverse)
{
	std::vector<degree_vertex_pair> d_prime(d_reverse);
	switch(d_prime.size()){
		case 1:
			return total;
		case 2:
		case 3:
			return total + 1;
		default:
			std::vector<degree_vertex_pair>::iterator it;
			d_prime.pop_back();
			d_prime.pop_back();
			total = NumberOfKGroupings(total, d_prime);
			while(d_prime.size() > 2){
				d_prime.pop_back();
				total = NumberOfKGroupings(total, d_prime);
			}
			total++;
			return total;
	}
}

///////////////////////////////////////////////////////////////////////////////////
//
// Return degree anonymizition cost when start, start+1, ..., end, are put in the 
//	same anonymized group
//
size_t DAGroupCost(std::vector<degree_vertex_pair> d, size_t start, size_t end)
{
	size_t total = 0;
	for(size_t i = start; i <= end; i++){
		total += d.at(start-1).first - d.at(i-1).first;
	}
	return total;
}

///////////////////////////////////////////////////////////////////////////////////
//
// Find the degree sequence (or all possible degree sequences if AllPossible = true, does not eliminate repeats)
//	of the graph
//
std::vector< std::vector<degree_vertex_pair> > DegreeAnonymization(std::vector<degree_vertex_pair> d, size_t number_of_vertices, size_t k, bool AllPossible = false)
{
	if(!AllPossible){
		std::vector<size_t> DA(number_of_vertices,0);
		std::vector<size_t> I(number_of_vertices,0);
		std::vector<degree_vertex_pair> DegreeSequence(d);
		std::vector<size_t> Split(number_of_vertices,0);
		std::vector< std::vector<degree_vertex_pair> > SetOfDegreeSequences;
		
		for(size_t i = 1; i <= number_of_vertices; i++){
			if( i < 2*k ){
				if( i > 1 ){
					I[i-1] += I[i-2] + d.at(0).first - d.at(i-1).first;
					DA[i-1] = I.at(i-1);
				}
			}
			else{
				size_t min = std::numeric_limits<size_t>::max();
				size_t t_opt = 0;
				for(size_t t = std::max(k,i-2*k+1); t <= i-k; t++){
					size_t cost = DA[t-1] + DAGroupCost(d,t+1,i);
					if( cost < min ){
						min = cost;
						t_opt = t;
					}
				}
				//for(size_t j = t_opt+1; j <= i; j++){
				//	DegreeSequence[j-1] = d.at(t_opt);
				//}
				Split[i-1] = t_opt;
				DA[i-1] = min;
			}
		}

		// Find anonymized degree sequence
		size_t previous_split = number_of_vertices;
		while( previous_split > 0 ){
			size_t current_split = Split[previous_split-1];
			if( d.at(previous_split-1).first != d.at(current_split).first ){
				for(size_t i = previous_split; i > current_split + 1; i--){
					DegreeSequence[i-1].first = d.at(current_split).first;
				}
			}
			previous_split = current_split;
		}

		// print results
		std::vector<degree_vertex_pair>::iterator it;
		log_file_verbose << "Cost of anonymizing: " << DA[number_of_vertices - 1] << std::endl;

		SetOfDegreeSequences.push_back(DegreeSequence);
		return SetOfDegreeSequences;
	}
	else{
		std::vector<size_t> DA(number_of_vertices,0);
		std::vector<size_t> I(number_of_vertices,0);
		std::vector<degree_vertex_pair> DegreeSequence(number_of_vertices);
		std::vector< std::vector<degree_vertex_pair> > SetOfPossibleDegreeSequences;
		std::vector< std::vector< std::vector<degree_vertex_pair> > > SetOfDegreeSequences;

		for(size_t i = 1; i <= number_of_vertices; i++){
			SetOfPossibleDegreeSequences.clear();
			if( i < 2*k ){
				DegreeSequence = d;
				if( i > 1 ){
					I[i-1] += I[i-2] + d.at(0).first - d.at(i-1).first;
					DA[i-1] = I.at(i-1);
				}
				for(size_t j = 1; j <= i; j++){
					DegreeSequence[j-1] = d.at(0);
				}
				SetOfPossibleDegreeSequences.push_back(DegreeSequence);
			}
			else{
				size_t min = std::numeric_limits<size_t>::max();
				for(size_t t = std::max(k,i-2*k+1); t <= i-k; t++){
					size_t cost = DA[t-1] + DAGroupCost(d,t+1,i);
					if( cost <= min ){
						if(cost != min){
							SetOfPossibleDegreeSequences.clear();
							min = cost;
						}
						std::vector< std::vector<degree_vertex_pair> >::iterator it;
						for(it = SetOfDegreeSequences[t-1].begin(); it < SetOfDegreeSequences[t-1].end(); ++it ){
							DegreeSequence = *it;
							for(size_t j = t+1; j <= i; j++){
								DegreeSequence[j-1].first = d.at(t).first;
							}
							SetOfPossibleDegreeSequences.push_back(DegreeSequence);
						}
					}
				}
				DA[i-1] = min;
			}
			SetOfDegreeSequences.push_back(SetOfPossibleDegreeSequences);
		}

		log_file_verbose << "Cost of anonymizing: " << DA[number_of_vertices - 1] << std::endl;
		//std::vector<degree_vertex_pair>::iterator it;
		//std::vector< std::vector<degree_vertex_pair> >::iterator it2;
		//for(it2 = SetOfDegreeSequences.back().begin(); it2 < SetOfDegreeSequences.back().end(); it2++ ){
		//	log_file_verbose << "Anonymized degree sequence: ";
		//	for(it = (*it2).begin(); it < (*it2).end()-1; ++it){
		//		log_file_verbose << (*it).first << ",";
		//	}
		//	log_file_verbose << (*it).first << std::endl;
		//}
		return SetOfDegreeSequences.back();
	}
}