#include <ctime>
#include <iostream>
#include <iterator>
#include <list>
#include <algorithm>
#include <limits>
#include <string>
#include <fstream>

#include <boost/algorithm/string.hpp>
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
#include <boost/graph/small_world_generator.hpp>
#include <boost/graph/max_cardinality_matching.hpp>
#include <boost/random.hpp>
#include <boost/timer.hpp>

#include "max_cardinality_implicit_matching.hpp"

using namespace std;
using namespace boost;

// Create random function which outputs from [0,1)
mt19937 rng( static_cast<unsigned int>(time(0)) );
static uniform_01<mt19937> uni01(rng);

// Graph definitions
struct vertex_info {};
typedef property<vertex_index_t, size_t> vertex_prop;

struct edge_info {};
typedef property<edge_index_t, size_t> edge_prop;

struct graph_info {};
typedef property<graph_name_t, string> graph_prop;

typedef adjacency_list<vecS, vecS, undirectedS, vertex_prop, edge_prop, graph_prop> Graph;
typedef subgraph<Graph> SubGraph;
typedef small_world_iterator<minstd_rand, SubGraph> SmallWorldGenerator;

typedef graph_traits<Graph>::vertex_descriptor vertex_descriptor_t;
typedef graph_traits<Graph>::vertex_iterator vertex_iterator_t;
typedef graph_traits<Graph>::edge_descriptor edge_descriptor_t;
typedef graph_traits<Graph>::edge_iterator edge_iterator_t;
typedef graph_traits<Graph>::out_edge_iterator out_edge_iterator_t;
typedef graph_traits<Graph>::adjacency_iterator adjacency_iterator_t;
typedef pair<vertex_descriptor_t, vertex_descriptor_t> vertex_pair_t;

// Graph input choices
enum { EXIT, SMALL_WORLD_GRAPH, ENRON_GRAPH, KARATE_GRAPH };
string graph_titles [] = {"Exit", "Small-world Graph", "Enron Dataset Graph", "Karate Dataset Graph"};

// Default graph/problem values
size_t default_number_of_experiments = 1;
size_t default_number_of_vertices = 1000;
size_t default_k = 50;
double default_subset_X_percent = 0.30;
size_t default_k_nearest_neighbors = 50;
size_t default_input_graph = SMALL_WORLD_GRAPH;

// Degree sequence definition
typedef pair<size_t,vertex_descriptor_t> degree_vertex_pair;

// Compare for sort creating a descending order
bool compare_descending(degree_vertex_pair i, degree_vertex_pair j){ return i.first > j.first; }

// Output to cout and file log.txt at same time using boost tee
typedef boost::iostreams::tee_device< ostream, iostreams::stream<iostreams::file_sink> > tee_device;
typedef boost::iostreams::stream<tee_device> tee_stream;
iostreams::stream<iostreams::file_sink> log_file;
tee_device tee(cout, log_file);
tee_stream cout_and_log_file(tee);

// Auxillary Functions
void exit_program();
void select_input_graph(size_t& input_graph);
void get_inputs(size_t& number_of_experiments, size_t& k, double& subset_X_percent);
void get_inputs(size_t& number_of_experiments, size_t& k, double& subset_X_percent, size_t& number_of_vertices, size_t& k_nearest_neighbors);
void print_augmenting_path(const string& description, vector<vertex_descriptor_t> augmenting_path, Graph G_prime, map<vertex_descriptor_t, set<vertex_descriptor_t> > mates, bool output_to_cout_and_error_check = true );
void print_t_degree_sequence(const string& description, vector<size_t> d);
void print_degree_sequence(const string& description, vector<degree_vertex_pair> d);
void print_degree_sequence(const string& description, map<vertex_descriptor_t, ptrdiff_t> d, bool output_to_cout = false);
void print_degree_sequence(const string& description, set<vertex_descriptor_t> d);
bool is_k_degree_anonymous(vector<degree_vertex_pair> degree_sequence, size_t k);
size_t ErdosGallaiThm(vector< vector<size_t> > SetOfDegreeSequences, size_t n);
size_t NumberOfKGroupings(size_t total, vector<size_t> d_reverse);
size_t DAGroupCost(vector<size_t> d, size_t start, size_t end);
vector< vector<degree_vertex_pair> > DegreeAnonymization(vector<degree_vertex_pair> d, size_t number_of_vertices, size_t k, bool AllPossible);
vector< vertex_pair_t > upper_degree_constrained_subgraph(Graph G, vector<degree_vertex_pair> d, map<vertex_descriptor_t, ptrdiff_t> upper_bounds, map<vertex_descriptor_t, ptrdiff_t> delta, size_t number_of_vertices_G_prime, size_t number_of_edges_G_prime, size_t number_of_vertices_G_prime_actual, size_t number_of_edges_G_prime_actual);
bool is_even( size_t value ){ return value % 2 == 0; }
bool is_odd( size_t value ){ return value % 2 == 1; }

// Matching functions
size_t greedy_implicit_initial_matching(Graph G_prime, map<vertex_descriptor_t, set<vertex_descriptor_t> > &mates, map<vertex_descriptor_t, ptrdiff_t> &num_exposed_subvertices);
size_t extra_greedy_implicit_initial_matching(Graph G_prime, map<vertex_descriptor_t, set<vertex_descriptor_t> > &mates, map<vertex_descriptor_t, ptrdiff_t> &num_exposed_subvertices);
vector<vertex_descriptor_t> edmonds_implicit_matching(Graph& G_prime, map<vertex_descriptor_t, set<vertex_descriptor_t> > mates, map<vertex_descriptor_t, ptrdiff_t> num_exposed_subvertices);
vector<vertex_descriptor_t> BFS_matching(Graph& G_prime, map<vertex_descriptor_t, set<vertex_descriptor_t> > mates, map<vertex_descriptor_t, ptrdiff_t> num_exposed_subvertices);
vector<vertex_descriptor_t> DFS_matching(Graph& G_prime, map<vertex_descriptor_t, set<vertex_descriptor_t> > mates, map<vertex_descriptor_t, ptrdiff_t> num_exposed_subvertices);
bool DFS_recursion(Graph& G_prime, vertex_descriptor_t current_vertex, bool currently_matched, vector<vertex_descriptor_t>& augmenting_path, set<vertex_descriptor_t> exposed_vertices, map<vertex_descriptor_t, ptrdiff_t> num_exposed_subvertices, map<edge_descriptor_t, bool>& marked_edges, map<vertex_descriptor_t, set<vertex_descriptor_t> > mates );
vector<vertex_descriptor_t> edmonds_find_augmenting_path(vertex_descriptor_t u, vertex_descriptor_t v, vector<vertex_descriptor_t> parent);
bool is_matched(vertex_descriptor_t u, vertex_descriptor_t v, map<vertex_descriptor_t, set<vertex_descriptor_t> > mates);

// Debug variables/functions
timer t;		// Main level debug timer
timer t_l2;	// Level 2 debug timer (debug statements contained within main level debug statements)
size_t recursion_level = 1;
void DEBUG_START(string);
void DEBUG_END(string);
void DEBUG_START_L2(string);
void DEBUG_END_L2(string);
void DEBUG_RECURSION_START(string);
void DEBUG_RECURSION_END(string);


int main()
{
	size_t number_of_experiments = 1;
	size_t number_of_vertices = 0;
	size_t k = 0;
	size_t k_nearest_neighbors = 0;
	size_t input_graph = SMALL_WORLD_GRAPH;
	double subset_X_percent = 0;

	while(1){
	
		// Name output file
		posix_time::time_facet *facet = new posix_time::time_facet("%Y-%b-%d_%Hh-%Mm-%Ss");
		stringstream ss;
		ss.str("");
		ss.imbue( locale(ss.getloc(), facet) );
		posix_time::ptime current_date_and_time = posix_time::second_clock::local_time();
		ss << "log_" << current_date_and_time << ".txt";
		string log_file_name(ss.str());
		if( log_file.is_open() ){
			log_file.close();
		}
		log_file.open(log_file_name);

		// Select type of graph
		select_input_graph(input_graph);

		///////////////////////////////////////////////////////////////////////////////////
		//
		// Create a graph, G
		//
		switch( input_graph ){
			case SMALL_WORLD_GRAPH:
				get_inputs(number_of_experiments, k, subset_X_percent, number_of_vertices, k_nearest_neighbors);
				break;

			case ENRON_GRAPH:
				get_inputs(number_of_experiments, k, subset_X_percent);
				number_of_vertices = 36692;
				break;

			case KARATE_GRAPH:
				get_inputs(number_of_experiments, k, subset_X_percent);
				number_of_vertices = 34;
				break;

			default:
				cout_and_log_file << "Error: no graph option selected?" << endl;
		}

		// Keep track of the current experiment number and how many times the experiments have succeeded and failed
		size_t current_experiment = 1;
		size_t successes = 0, failures = 0;

		// Start looping until number_of_experiments have been run
		while( current_experiment <= number_of_experiments ){
			cout_and_log_file << endl << endl;
			cout_and_log_file << "-----------------------------------------" << endl;
			cout_and_log_file << "Experiment number: " << current_experiment << endl;

			SubGraph G(number_of_vertices);

			switch( input_graph ){
				case SMALL_WORLD_GRAPH:
					{
					///////////////////////////////////////////////////////////////////////////////////
					//
					// Create a small-world graph (http://www.boost.org/doc/libs/1_48_0/libs/graph/doc/small_world_generator.html)
					//	small_world_iterator(RandomGenerator& gen, vertices_size_type n,
					//                    vertices_size_type k, double probability,
					//                    bool allow_self_loops = false);
					//	Constructs a small-world generator iterator that creates a graph with n vertices, each connected to its k nearest neighbors. Probabilities are drawn from the random number generator gen. Self-loops are permitted only when allow_self_loops is true.
					//
					
					DEBUG_START("Creating small-world graph ...");
					minstd_rand gen;
					bool allow_self_loops = false;
					double probability = uni01();
					cout_and_log_file << "Number of vertices: " << number_of_vertices << endl;
					cout_and_log_file << "Each connected to its " << k_nearest_neighbors << " nearest neighbors." << endl;
					cout_and_log_file << "Edges in the graph are randomly rewired to different vertices with a probability " << probability << " and self-loops are set to " << allow_self_loops << "." << endl;
					SmallWorldGenerator smg_it;
					for(smg_it = SmallWorldGenerator(gen, number_of_vertices, k_nearest_neighbors, probability, allow_self_loops); smg_it != SmallWorldGenerator(); ++smg_it){
						add_edge( (*smg_it).first, (*smg_it).second, G );
					}
					DEBUG_END("Creating small-world graph ...");
					}
					break;

				case ENRON_GRAPH:
					{
					// Enron graph properties
					size_t number_of_edges = 367662;

					// Read in Enron data
					DEBUG_START("Reading in enron data ...");
					vector< pair<size_t,size_t> > edge_list( number_of_edges / 2);
					string line;
					iostreams::stream<iostreams::file_source> input_file("Email-Enron.txt");
					size_t number_of_edges_read = 0;
					if(input_file.is_open()){
						while(getline(input_file, line)){
							if(line.at(0) != '#'){
								vector<string> tokens;
								split(tokens, line, is_any_of("\t"));
								size_t u = atoi(tokens.at(0).c_str());
								size_t v = atoi(tokens.at(1).c_str());
								if( u < v ){
									edge_list[number_of_edges_read++] = make_pair(u, v);
									if( number_of_edges_read % 10000 == 0 ){
										cout_and_log_file << "Read " << number_of_edges_read << " edges out of 183831." << endl;
									}
								}
							}
						}
						input_file.close();
					}
					else{
						cout_and_log_file << "Unable to open file" << endl;
					}
					DEBUG_END("Reading in enron data ...");

					DEBUG_START("Creating graph from enron data ...");
					vector< pair<size_t,size_t> >::iterator eit;
					for(eit = edge_list.begin(); eit < edge_list.end(); ++eit){
						add_edge( (*eit).first, (*eit).second, G );
					}
					DEBUG_END("Creating graph from enron data ...");

					// Write Enron data in graphml format
					//dynamic_properties dp;
					//iostreams::stream<iostreams::file_sink> output_file("Enron Email Data.graphml");
					//DEBUG_START("Writing enron data to file Enron Email Data.graphml ...");
					//write_graphml(output_file, G, dp);
					//DEBUG_END("Writing enron data to file Enron Email Data.graphml ...");
					//output_file.close();

					// Read in Enron data in graphml format
					//Graph G;
					//iostreams::stream<iostreams::file_source> input_file("Enron Email Data.graphml");
					//dynamic_properties dp;
					//property_map<Graph, vertex_index_t>::type node_id_map = get(vertex_index, G);
					//dp.property("node_id", node_id_map);
					//DEBUG_START("Reading in enron data ...");
					//read_graphml(input_file, G, dp);
					//DEBUG_END("Reading in enron data ...");
					}
					break;

				case KARATE_GRAPH:
					{
					// Karate graph properties
					size_t number_of_edges = 78;

					// Read in Karate data
					DEBUG_START("Reading in karate data ...");
					vector< pair<size_t,size_t> > edge_list( number_of_edges );
					string line;
					iostreams::stream<iostreams::file_source> input_file("karate.txt");
					size_t number_of_edges_read = 0;
					if(input_file.is_open()){
						while(getline(input_file, line)){
							if(line.at(0) != '#'){
								vector<string> tokens;
								split(tokens, line, is_any_of("\t"));
								size_t u = atoi(tokens.at(0).c_str());
								size_t v = atoi(tokens.at(1).c_str());
								edge_list[number_of_edges_read++] = make_pair(u, v);
							}
						}
						input_file.close();
					}
					else{
						cout_and_log_file << "Unable to open file" << endl;
					}
					DEBUG_END("Reading in karate data ...");

					DEBUG_START("Creating graph from karate data ...");
					vector< pair<size_t,size_t> >::iterator eit;
					for(eit = edge_list.begin(); eit < edge_list.end(); ++eit){
						add_edge( (*eit).first, (*eit).second, G );
					}
					DEBUG_END("Creating graph from karate data ...");
					}
					break;

				default:
					cout_and_log_file << "Error: no graph option selected?" << endl;
			}

			/*
			///////////////////////////////////////////////////////////////////////////////////
			//
			// Create a random graph with number_of_vertices vertices
			//

			Graph G(number_of_vertices);
			
			// Add edges
			double edge_probability = uni01();
			vertex_iterator ui, ui_end, vi;
			for(tie(ui, ui_end) = vertices(G); ui != ui_end; ++ui){
				for(vi = ui+1; vi != ui_end; ++vi){
					// Use for random graphs
					double rand1 = uni01();
					if( rand1 <= edge_probability )
						add_edge(*ui, *vi, G);
				}
			}
			*/
			//DEBUG_START("Writing graph G to graph file in graphviz .dot format ...");
			//ss.str("");
			//ss << "graph_" << current_date_and_time << ".dot";
			//iostreams::stream<iostreams::file_sink> graph_file(ss.str());
			//write_graphviz(graph_file, G);
			//DEBUG_END("Writing graph G to graph file in graphviz .dot format ...");

			///////////////////////////////////////////////////////////////////////////////////
			//
			// Find a small subset of vertices, X, of G
			//
			DEBUG_START("Creating subset X of G ...");
			set<vertex_descriptor_t> X_vertices;
			if( subset_X_percent <= 0.9999999 ){
				while(X_vertices.size() < subset_X_percent * num_vertices(G)){
					// randomly add vertices to X
					double rand1 = uni01();
					size_t add_vertex_num = (size_t)(rand1 * num_vertices(G));
					X_vertices.insert( vertex(add_vertex_num, G) );
				}
				print_degree_sequence("Vertices chosen for X from G", X_vertices);
			}
			else{
				vertex_iterator_t ui, ui_end, vi;
				for(tie(ui, ui_end) = vertices(G); ui != ui_end; ++ui){
					X_vertices.insert(*ui);
				}
			}

			SubGraph& X_sub = G.create_subgraph(X_vertices.begin(), X_vertices.end());
			Graph X;
			copy_graph(X_sub, X);
			size_t number_of_vertices_X = num_vertices(X);
			cout_and_log_file << "Number of vertices in X: " << number_of_vertices_X << endl;
			DEBUG_END("Creating subset X of G ...");

			// Find degree sequence
			vector< degree_vertex_pair > d;
			vertex_iterator_t ui, ui_end, vi;
			for(tie(ui, ui_end) = vertices(X); ui != ui_end; ++ui){
				//degree_vertex_pair degree_vertex( degree( *ui,X ), *ui ); // Anonymize induced X
				degree_vertex_pair degree_vertex( degree(X_sub.local_to_global(*ui), G), *ui ); // Anonymize X wrt G
				d.push_back(degree_vertex);
			}
			sort(d.begin(), d.end(), compare_descending);
			print_degree_sequence("Degree sequence", d);
			
			// Check for clique
			if( d.front().first == d.back().first ){
				cout_and_log_file << "Clique or no edges, skipping." << endl;
				continue;
			}

			/*
			// Determine total number of k-groupings
			size_t total = 0;
			vector<size_t> d_reverse(d.rbegin(), d.rend());
			total = NumberOfKGroupings(total, d_reverse);
			cout_and_log_file << "The total number of k-groupings: " << total << endl;
			*/
			

			///////////////////////////////////////////////////////////////////////////////////
			//
			// Degree Anonymization (Lui and Terzi 2008)
			//
			DEBUG_START("Determining degree anonymization ...");
			vector< vector<degree_vertex_pair> > SetOfDegreeSequences;
			SetOfDegreeSequences = DegreeAnonymization(d, number_of_vertices_X, k, false);
			vector<degree_vertex_pair> AnonymizedDegreeSequence(SetOfDegreeSequences.back());
			print_degree_sequence("Anonymized degree sequence", AnonymizedDegreeSequence);
			DEBUG_END("Determining degree anonymization ...");

			// Check if it is a real degree sequence
			// ErdosGallaiThm(SetOfDegreeSequences.back(), number_of_vertices);

			///////////////////////////////////////////////////////////////////////////////////
			//
			// Find lower and upper bounds and delta
			//
			DEBUG_START("Determine the upper bounds for vertices in X");
			map<vertex_descriptor_t, ptrdiff_t> upper_bounds;
			vector<degree_vertex_pair>::iterator dvit;
			size_t anonymized_degree_sequence_index = 0;
			size_t number_of_vertices_X_prime = 0;
			size_t number_of_edges_X_prime = 0; // num_edges(X_com); // if including only internal to external edges, use 0, for all edges use num_edges(X_com)
			for(dvit = d.begin(); dvit != d.end(); ++dvit){
				vertex_descriptor_t u = (*dvit).second;
				size_t d_i = number_of_vertices_X - 1 - degree(u, X);
				ptrdiff_t u_i = AnonymizedDegreeSequence.at(anonymized_degree_sequence_index++).first - (*dvit).first;
				upper_bounds[u] = u_i;
				ptrdiff_t delta_i = d_i - u_i;

				// Determine the number of edges that would be used if vertices with u_i == 0 weren't excluded
				number_of_vertices_X_prime += d_i + delta_i;
				number_of_edges_X_prime += d_i * delta_i;
			}
			print_degree_sequence("Upper bounds", upper_bounds);
			DEBUG_END("Determine the upper bounds for vertices in X");


			// Find the complement of the subset X
			DEBUG_START("Finding complement of X ...");
			//size_t number_of_edges_read_X = 0;
			//vector< pair<vertex_descriptor_t,vertex_descriptor_t> > edge_list_X_com( number_of_vertices_X * (number_of_vertices_X - 1) / 2 - num_edges(X) );
			Graph X_com( number_of_vertices_X );
			for(tie(ui, ui_end) = vertices(X); ui != ui_end; ++ui){
				for(vi = ui+1; vi != ui_end; ++vi){
					// Only add edge if not in X AND it's upper bounds of both vertices is greater than 0 (ignoring any edges that cannot be in matching)
					if( !edge(*ui, *vi, X).second && upper_bounds[*ui] > 0 && upper_bounds[*vi] > 0 ){
						//edge_list_X_com[number_of_edges_read_X++] = make_pair(*ui, *vi);
						add_edge(*ui,*vi,X_com);
					}
				}
			}
			//Graph X_com(edge_list_X_com.begin(), edge_list_X_com.end(), number_of_vertices_X);
			DEBUG_END("Finding complement of X ...");

			DEBUG_START("Determine the lower bounds and delta values for every vertex in X/X_com");
			map<vertex_descriptor_t, ptrdiff_t> lower_bounds, delta;
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
					number_of_vertices_X_prime_actual += d_i + delta_i;
					number_of_edges_X_prime_actual += d_i * delta_i;
				}
			}
			print_degree_sequence("Lower bounds", lower_bounds);
			print_degree_sequence("Delta values", delta);
			DEBUG_END("Determine the upper, lower bounds and delta values for every vertex in X/X_com");

			///////////////////////////////////////////////////////////////////////////////////
			//
			// Create X_star
			//

			///////////////////////////////////////////////////////////////////////////////////
			//
			// Find upper degree-constrained subgraph (H_star) on X_star
			//
			vector< vertex_pair_t > H_star = upper_degree_constrained_subgraph(X_com, d, upper_bounds, delta, number_of_vertices_X_prime, number_of_edges_X_prime, number_of_vertices_X_prime_actual, number_of_edges_X_prime_actual);

			DEBUG_START("Adding edges from H to X, displaying final degree sequence ...");
			vector< pair<vertex_descriptor_t,vertex_descriptor_t> >::iterator vvit;
			log_file << "Edges to add to X to make it k-degree anonymous:" << endl << endl;
			log_file << "Global\t\tLocal" << endl;
			for(vvit = H_star.begin(); vvit < H_star.end(); ++vvit){
				log_file << "(" << X_sub.local_to_global( (*vvit).first ) << "," << X_sub.local_to_global( (*vvit).second ) << ")\t\t(" << (*vvit).first << "," << (*vvit).second << ")" << endl;
				add_edge( vertex((*vvit).first, X_sub), vertex((*vvit).second, X_sub), X_sub); // Add edge to X
				upper_bounds[(*vvit).first]--; 	// Determine upper and lower bounds
				upper_bounds[(*vvit).second]--; 
				if( lower_bounds[(*vvit).first] > 0 )	
					lower_bounds[(*vvit).first]--;
				if( lower_bounds[(*vvit).second] > 0 )			
					lower_bounds[(*vvit).second]--;
			}

			// Find degree sequence of X_final
			vector<degree_vertex_pair> d_added_edges_within_X;
			for(tie(ui, ui_end) = vertices(X_sub); ui != ui_end; ++ui){
				//degree_vertex_pair degree_vertex( degree( *ui,X_final ), *ui ); // Anonymize induced X
				degree_vertex_pair degree_vertex( degree(X_sub.local_to_global(*ui), G), *ui ); // Anonymize X wrt G
				d_added_edges_within_X.push_back(degree_vertex);
			}
			sort(d_added_edges_within_X.begin(), d_added_edges_within_X.end(), compare_descending);
			print_degree_sequence("Degree sequence of graph after added edges within X", d_added_edges_within_X);
			print_degree_sequence("Upper bounds (final)", upper_bounds);
			print_degree_sequence("Lower bounds (final)", lower_bounds);
			DEBUG_END("Adding edges from H to X, displaying final degree sequence ...");

			DEBUG_START("Find possible additional edges in G \\ X to make X k-anonymous...");
			// Find which vertices need more edges to become anonymized
			vector<degree_vertex_pair> unanonymized_vertices;
			size_t i = 0;
			for(dvit = d_added_edges_within_X.begin(); dvit < d_added_edges_within_X.end(); ++dvit){
				ptrdiff_t u_i = AnonymizedDegreeSequence.at(i++).first - (*dvit).first;
				if( u_i > 0 ){
					unanonymized_vertices.push_back( make_pair(u_i, (*dvit).second) );
				}
				else if( u_i < 0 ){
					cout_and_log_file << "ERROR!!! Added too many edges to X such that the upper bounds of an edge exceeded the anonymized degree sequence." << endl;
					cout_and_log_file << "Occured on vertex: " << (*dvit).second << endl;
					cout_and_log_file << "Vertex degree: " << (*dvit).first << endl;
					cout_and_log_file << "Anonymized degree value: " << AnonymizedDegreeSequence.at(i-1).first << endl;
					cout_and_log_file << "Upper bound: " << u_i << endl;

					DEBUG_START("Writing graph G to graph file in graphviz .dot format ...");
					ss.str("");
					ss << "graph_" << current_date_and_time << ".dot";
					iostreams::stream<iostreams::file_sink> graph_file(ss.str());
					write_graphviz(graph_file, G);
					DEBUG_END("Writing graph G to graph file in graphviz .dot format ...");
					
					exit_program();
				}
			}

			// If there are any unanonymized vertices, find possible edges to make X k-anonymous, otherwise exit noting success
			bool edges_found = true;
			vector<degree_vertex_pair> d_final;
			if( unanonymized_vertices.size() > 0 ){
				sort(unanonymized_vertices.begin(), unanonymized_vertices.end(), compare_descending);

				log_file << "Edges to add to X to make it k-degree anonymous (in X, in G \\ X):" << endl << endl;

				// Find list of all possible nodes in G which can be connected to an unanonymized vertex in X
				size_t total_num_of_added_edges = 0;
				for(dvit = unanonymized_vertices.begin(); dvit < unanonymized_vertices.end(); ++dvit){
					size_t upper_bound = (*dvit).first;

					vertex_descriptor_t v = (*dvit).second;
					vertex_descriptor_t v_global = X_sub.local_to_global(v);
					vector<vertex_descriptor_t> possible_vertices;
					for(tie(ui, ui_end) = vertices(G); ui != ui_end; ++ui){
						if( !edge(*ui, v_global, G).second && !X_sub.find_vertex(*ui).second ){	// Check that there's no edge from u to v and u is not in X
							possible_vertices.push_back(*ui);
						}
					}

					size_t num_of_added_edges = 0;
					while(num_of_added_edges < upper_bound){
						// Randomly add edges from G to X for unanonymized vertices in X
			
						double rand1 = uni01();
						if( possible_vertices.size() > 0 ){
							size_t random_index = (size_t)(rand1 * possible_vertices.size());
							log_file << "(" << v_global << "," << possible_vertices.at(random_index) << ")" << endl;
							add_edge(vertex(possible_vertices.at(random_index), G), v_global, G);
							possible_vertices.erase(possible_vertices.begin() + random_index);
							upper_bounds[v]--; 
							if( lower_bounds[v] > 0 )
								lower_bounds[v]--; 
							num_of_added_edges++;
						}
						else{
							edges_found = false;
							break;
						}
					}
					total_num_of_added_edges += num_of_added_edges;
				}

				cout_and_log_file << endl << "Number of edges added to X from G \\ X: " << total_num_of_added_edges << endl << endl;

				// Find degree sequence of X
				for(tie(ui, ui_end) = vertices(X_sub); ui != ui_end; ++ui){
					degree_vertex_pair degree_vertex( degree(X_sub.local_to_global(*ui), G), *ui ); // Anonymize X wrt G
					d_final.push_back(degree_vertex);
				}
				sort(d_final.begin(), d_final.end(), compare_descending);

				print_degree_sequence("Degree sequence of final graph X", d_final);
			}
			else{
				d_final = d_added_edges_within_X;
				cout_and_log_file << "No edges needed from G \\ X to X to make X k-anonymous." << endl;
			}

			print_degree_sequence("Upper bounds (final)", upper_bounds);
			print_degree_sequence("Lower bounds (final)", lower_bounds);
			DEBUG_END("Find possible edges to add from G \\ X to X to make X k-anonymous...");

			DEBUG_START("Check if X is k-anonymous...");
			if( is_k_degree_anonymous(d_final, k) ){ // if( edges_found ){
				cout_and_log_file << "SUCCESS: X made k-anonymous." << endl;
				successes++;
			}
			else{
				cout_and_log_file << "FAIL: X not made k-anonymous." << endl;

				DEBUG_START("Writing graph G to graph file in graphviz .dot format ...");
				ss.str("");
				ss << "graph_" << current_date_and_time << ".dot";
				iostreams::stream<iostreams::file_sink> graph_file(ss.str());
				write_graphviz(graph_file, G);
				DEBUG_END("Writing graph G to graph file in graphviz .dot format ...");

				failures++;
			}
			DEBUG_END("Check if X is k-anonymous...");

			DEBUG_START("Check if G is k-anonymous...");
			vector<degree_vertex_pair> degree_sequence_G;
			for(tie(ui, ui_end) = vertices(G); ui != ui_end; ++ui){
				degree_vertex_pair degree_vertex( degree(*ui, G), *ui ); 
				degree_sequence_G.push_back(degree_vertex);
			}
			sort(degree_sequence_G.begin(), degree_sequence_G.end(), compare_descending);

			if( is_k_degree_anonymous(degree_sequence_G, k) ){
				cout_and_log_file << "G is k-anonymous." << endl;
			}
			else{
				cout_and_log_file << "G is not k-anonymous." << endl;
			}
			DEBUG_END("Check if G is k-anonymous...");

			current_experiment++;
			cout_and_log_file << "Succeeded " << successes << " times and failed " << failures << " times." << endl;
		}
	}
}

vector< vertex_pair_t > upper_degree_constrained_subgraph(Graph G, vector<degree_vertex_pair> d, map<vertex_descriptor_t, ptrdiff_t> upper_bounds, map<vertex_descriptor_t, ptrdiff_t> delta, size_t number_of_vertices_G_prime, size_t number_of_edges_G_prime, size_t number_of_vertices_G_prime_actual, size_t number_of_edges_G_prime_actual){
	Graph G_prime;
	DEBUG_START("Initializing implicit representation ...");
	copy_graph(G, G_prime);
	number_of_vertices_G_prime_actual = num_vertices(G_prime);
	number_of_edges_G_prime_actual = num_edges(G_prime);

	map<vertex_descriptor_t, ptrdiff_t> num_exposed_subvertices;
	
	cout_and_log_file << "G' properties:" << endl;
	cout_and_log_file << "\tNumber of vertices: " << number_of_vertices_G_prime << endl;
	cout_and_log_file << "\tNumber of edges: " << number_of_edges_G_prime + num_edges(G) << endl; // Not quite right, as we erase edges adjacent to vertices with u_i == 0 in G before this
	cout_and_log_file << "\tNumber of vertices (actually used): " << number_of_vertices_G_prime_actual << endl;
	cout_and_log_file << "\tNumber of edges (actually used): " << number_of_edges_G_prime_actual + num_edges(G) << endl;	

	size_t total_anonymization_cost = 0;

	// Implicit representation
	vertex_iterator_t ui, ui_end;
	for(tie(ui, ui_end) = vertices(G_prime); ui != ui_end; ++ui){
		// Number of exposed subvertices for each severed vertex is initially the upper bounds of each vertex
		num_exposed_subvertices[*ui] = upper_bounds[*ui];
		total_anonymization_cost += upper_bounds[*ui];
	}
	DEBUG_END("Initializing implicit representation ...");
	
	vector< pair<vertex_descriptor_t,vertex_descriptor_t> > H;	// Contains degree-constrained subgraph
	DEBUG_START("Finding matching on G' ...");

	map<vertex_descriptor_t, set<vertex_descriptor_t> > mates;	// List of all possible mates in implicit representation of a vertex
	map<vertex_descriptor_t, set<vertex_descriptor_t> > mates_extra_greedy;	// List of all possible mates from greedy initial matching
	map<vertex_descriptor_t, set<vertex_descriptor_t> > mates_greedy;	// List of all possible mates from extra greedy initial matching
	
	//edmonds_implicit_maximum_cardinality_matching(G_prime, &mates[0], &num_exposed_subvertices[0]);

	DEBUG_START_L2("Implicit representation: Finding initial matching ...");
	map<vertex_descriptor_t, ptrdiff_t> num_exposed_subvertices_extra_greedy(num_exposed_subvertices);
	map<vertex_descriptor_t, ptrdiff_t> num_exposed_subvertices_greedy(num_exposed_subvertices);
	size_t initial_anonymized_cost_handled_extra_greedy = extra_greedy_implicit_initial_matching(G_prime, mates_extra_greedy, num_exposed_subvertices_extra_greedy);
	size_t initial_anonymized_cost_handled_greedy = greedy_implicit_initial_matching(G_prime, mates_greedy, num_exposed_subvertices_greedy);

	cout_and_log_file << "\tTotal anonymization cost: " << total_anonymization_cost << endl;
	cout_and_log_file << "\tTotal anonymization cost handled by initial matching (greedy): " << initial_anonymized_cost_handled_greedy << endl;
	cout_and_log_file << "\tTotal anonymization cost handled by initial matching (extra greedy): " << initial_anonymized_cost_handled_extra_greedy << endl;

	size_t initial_anonymized_cost_handled = 0;
	if( initial_anonymized_cost_handled_extra_greedy > initial_anonymized_cost_handled_greedy ){
		mates = mates_extra_greedy;
		num_exposed_subvertices = num_exposed_subvertices_extra_greedy;
		initial_anonymized_cost_handled = initial_anonymized_cost_handled_extra_greedy;
		cout_and_log_file << "\tUsing extra greedy initial matching." << endl;
	}
	else{
		mates = mates_greedy;
		num_exposed_subvertices = num_exposed_subvertices_greedy;
		initial_anonymized_cost_handled = initial_anonymized_cost_handled_greedy;
		cout_and_log_file << "\tUsing greedy initial matching." << endl;
	}
	cout_and_log_file << "\tCost remaining: " <<  total_anonymization_cost - initial_anonymized_cost_handled << endl;
	cout_and_log_file << "\tMax number of augmenting paths to find: " <<  (total_anonymization_cost - initial_anonymized_cost_handled) / 2 << endl;
	print_degree_sequence("Number of exposed subvertices", num_exposed_subvertices, true);
	DEBUG_END_L2("Implicit representation: Finding initial matching ...");

	size_t augmenting_path_number = 0;	// Keeps track of number of times edmonds_implicit_matching is call (does not include recursive calls)
	// Use BFS to find paths from exposed vertex to exposed vertex, where the exposed vertices are different, then use DFS to find when they are the same
	DEBUG_START_L2("Find augmenting path (BFS) ...");
	vector<vertex_descriptor_t> augmenting_path = edmonds_implicit_matching(G_prime, mates, num_exposed_subvertices);
	cout_and_log_file << "\tAugmenting path number " << ++augmenting_path_number << " found ..." << endl;
	DEBUG_END_L2("Find augmenting path (BFS) ...");

	while( !augmenting_path.empty() ){
		// Reverse matching on augmenting path
		bool change_to_matching = true;
		vector<vertex_descriptor_t>::iterator it;
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
		print_degree_sequence("Number of exposed subvertices", num_exposed_subvertices, true);

		DEBUG_START_L2("Find augmenting path (BFS) ...");
		augmenting_path = edmonds_implicit_matching(G_prime, mates, num_exposed_subvertices);
		if( !augmenting_path.empty() ){
			cout_and_log_file << "\tAugmenting path number " << ++augmenting_path_number << " found ..." << endl;
		}
		DEBUG_END_L2("Find augmenting path (BFS) ...");
	}

	DEBUG_START_L2("Find augmenting path (DFS) ...");
	augmenting_path = edmonds_implicit_matching(G_prime, mates, num_exposed_subvertices);
	cout_and_log_file << "\tAugmenting path number " << ++augmenting_path_number << " found ..." << endl;
	DEBUG_END_L2("Find augmenting path (DFS) ...");

	while( !augmenting_path.empty() ){
		// Reverse matching on augmenting path
		bool change_to_matching = true;
		vector<vertex_descriptor_t>::iterator it;
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
		print_degree_sequence("Number of exposed subvertices", num_exposed_subvertices, true);

		DEBUG_START_L2("Find augmenting path (DFS) ...");
		augmenting_path = edmonds_implicit_matching(G_prime, mates, num_exposed_subvertices);
		if( !augmenting_path.empty() ){
			cout_and_log_file << "\tAugmenting path number " << ++augmenting_path_number << " found ..." << endl;
		}
		DEBUG_END_L2("Find augmenting path (DFS) ...");
	}

	DEBUG_START_L2("Set edges to be added to G ...");
	total_anonymization_cost = 0;
	for(tie(ui,ui_end) = vertices(G_prime); ui != ui_end; ++ui){
		set<vertex_descriptor_t>::iterator sit;
		for(sit = mates[*ui].begin(); sit != mates[*ui].end(); ++sit){
			if(*ui < *sit){
				total_anonymization_cost += 2;
				H.push_back( make_pair(*ui, *sit) );
			}
		}
	}
	cout_and_log_file << "\tActual cost of anonymizing handled in X only: " << total_anonymization_cost << endl;
	cout_and_log_file << "\tNumber of edges added to X: " << H.size() << endl;
	DEBUG_END_L2("Set edges to be added to G ...");

	DEBUG_END("Finding matching on G' ...");

	return H;
}

vector<vertex_descriptor_t> edmonds_implicit_matching(Graph& G_prime, map<vertex_descriptor_t, set<vertex_descriptor_t> > mates, map<vertex_descriptor_t, ptrdiff_t> num_exposed_subvertices){
	
	vector<vertex_descriptor_t> augmenting_path;
	Graph G_prime_with_contractions(G_prime);

	///////////////////////////////////////////////////////////////////////////////////
	//
	// Initial data structures to keep track of information for implicit matching
	//
	map< vertex_descriptor_t, vector<vertex_descriptor_t> > blossoms;
	vector<vertex_descriptor_t> parent( num_vertices(G_prime), graph_traits<Graph>::null_vertex() );
	vector<vertex_descriptor_t> root( num_vertices(G_prime), graph_traits<Graph>::null_vertex() );
	vector<ptrdiff_t> level( num_vertices(G_prime), -1 );	// Level of vertex in F
	vector<vertex_descriptor_t> in_blossom( num_vertices(G_prime), graph_traits<Graph>::null_vertex() );
	vector<edge_descriptor_t> even_edges;
	vector<bool> marked_edges( num_edges(G_prime), false );


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

			// Add edges adjacent to exposed vertices to even_edges
			out_edge_iterator_t ei, ei_end;
			for(boost::tie(ei,ei_end) = out_edges(v, G_prime); ei != ei_end; ++ei){
				if( !is_matched( source(*ei, G_prime), target(*ei, G_prime), G_prime) ){
					even_edges.insert(*ei);
					marked_edges.at(*ei) = true;
				}
			}
		}
	}

	cout_and_log_file << "\tedmonds_implicit_matching:  Total number of exposed subvertices: " << total_num_exposed_subvertices << endl;
	if( total_num_exposed_subvertices <= 1 ){
		return augmenting_path;
	}

	
	bool augmenting_path_found = false;
	vertex_descriptor_t v, w, w_original;
	while( !even_edges.empty() && !augmenting_path_found ){
		edge_descriptor_t current_edge = even_edges.back();
		even_edges.pop_back();

		v = source(current_edge, G_prime_with_contractions);
		w_original = target(current_edge, G_prime_with_contractions);	// Original vertex, before contraction
		w = target(current_edge, G_prime_with_contractions);

		// Check if w is in contracted blossom, if so, change to contracted blossom vertex
		if( in_blossom[w] != graph_traits<Graph>::null_vertex() ){
			w = in_blossom[w];
		}

		// Level of v is odd, should be even, so swap v and w
		if( is_odd(level[v]) ){	
			swap(v,w);
		}

		// Check if edge has already been checked


		if( root[w] == graph_traits<Graph>::null_vertex() ){ // True if w hasn't been added to forest
			level[w] = level[v] + 1;

			set<vertex_descriptor_t>::iterator it;
			for(it = mate[w_prime].begin(); it != mate[w_prime].end(); ++it){
				if( root[*it] == graph_traits<Graph>::null_vertex()){ // True if current match of w (*it) hasn't been added to forest
					level[*it] = level[w] + 1;
					parent[*it] = w;
					root[*it] = root[w];
					out_edge_iterator_t ei, ei_end;
					for( boost::tie(ei,ei_end) = out_edges(*it, G_prime_with_contractions); ei != ei_end; ++ei){	// Adding unmatched edges
						if( !marked_edges.at(*ei) && !is_matched( source(*ei, G_prime_with_contractions), target(*ei, G_prime_with_contractions), G_prime_with_contractions) ){
							even_edges.push_back(*ei);
							marked_edges.at(*ei) = true;
						}
					}
				}
			}
			
			parent[w] = v;
			root[w] = root[v];
		}
		else if( root[v] != root[w] ){	// If they have differing roots, it's an augmenting path
			augmenting_path_found = true;
		}
		else if( is_even(level[w]) ){	// Is a blossom
			// Find blossom tip
			vertex_descriptor_t blossom_tip = find_blossom_tip(v, w, root, parent);
			log_file << "Blossom tip: " << blossom_tip << endl;

			// Check if tip of blossom is exposed and has 2 or more exposed subvertices.  If so, it forms an augmenting path, otherwise its a blossom.
			if( num_exposed_subvertices[blossom_tip] >= 2 ){
				augmenting_path_found = true;
			}

			// Find blossom
			vector<vertex_descriptor_t> blossom;	// Tip of blossom listed at beginning and end
			vector<vertex_descriptor_t> augmenting_path_v_blossom_tip = edmonds_find_augmenting_path(v, blossom_tip, parent); // Finds path from v to blossom_tip
			vector<vertex_descriptor_t> augmenting_path_w_blossom_tip = edmonds_find_augmenting_path(w, blossom_tip, parent); // Finds path from w to blossom_tip
			blossom.assign( augmenting_path_v_blossom_tip.rbegin(), augmenting_path_v_blossom_tip.rend() );
			blossom.insert( blossom.end(), augmenting_path_w_blossom_tip.begin(), augmenting_path_w_blossom_tip.end() );
			print_augmenting_path("Blossom", blossom, G_prime_with_contractions, mates);

			vertex_descriptor_t contracted_blossom_vertex = add_vertex(G_prime_with_contractions);
			log_file << "Contracted blossom vertex number: " << blossom_vertex << endl;

			blossoms[contracted_blossom_vertex] = blossom;

			// Check if blossom has any vertex adjacent to it which is already in the forest, if so, it's an augmenting path
			// Contract at same time
			set<vertex_descriptor_t> blossom_adjacent_vertices;
			for(vector<vertex_descriptor_t>::iterator current_blossom_vertex = blossom.begin(); current_blossom_vertex < blossom.end()-1; ++current_blossom_vertex){	// Don't check start_blossom twice (blossom.end()-1)
				out_edge_iterator_t ei, ei_end;
				for(boost::tie(ei,ei_end) = out_edges(*current_blossom_vertex, G_prime_with_contractions); ei != ei_end; ++ei){
					vertex_descriptor_t x = target(*ei, G_prime_with_contractions);
					if( *current_blossom_vertex == target(*ei, G_prime_with_contractions) )
						x = source(*ei, G_prime_with_contractions);

					if( root[x] != graph_traits<Graph>::null_vertex() ){ // True if vertex adjacent to a blossom vertex is in forest
						// Augmenting path possibly exists
						if( root[x] != root[blossom_tip] || num_exposed_subvertices[root[v]] >= 2 ){
							v = x;
							w = blossom_tip;
							augmenting_path_found = true;
							break;
						}
					}

					// If blossom vertex was a mate to a vertex in G', set it to be a mate with the contracted blossom vertex
					if( mates[*blossom_vertex].find(x) != mates[*blossom_vertex].end() ){
						mates[x].insert(contracted_blossom_vertex);
						mates[contracted_blossom_vertex].insert(x);
					}
					blossom_adjacent_vertices.insert(x);	// Does not insert duplicates
				}

				if(augmenting_path_found)
					break;

				in_blossom[*current_blossom_vertex] = contracted_blossom_vertex;
				clear_vertex(*it, G_prime_with_contractions);	// Remove all edges from blossom vertex (only in G_prime_with_contractions)
			}

			for(set<vertex_descriptor_t>::iterator sit = blossom_adjacent_vertices.begin(); sit != blossom_adjacent_vertices.end(); ++sit){
				add_edge(blossom_vertex, *sit, G_prime_with_contractions);
			}

			// If the tip of the blossom was exposed, the blossom has 1 exposed subvertex (if it had two or more, would have been augmenting path), 0 otherwise
			num_exposed_subvertices[contracted_blossom_vertex] = num_exposed_subvertices[blossom_tip];
			root[contracted_blossom_vertex] = root[blossom_tip];
			parent[contracted_blossom_vertex] = parent[blossom_tip];
			level[contracted_blossom_vertex] = level[blossom_tip];
		}
	}

	if( !augmenting_path_found )
		return augmenting_path;

	// Find augmenting path
	vector<vertex_descriptor_t> augmenting_path_v = edmonds_find_augmenting_path(v, root[v], parent); // Finds path from v to root[v] (must insert reverse of this into augmenting_path)							
	vector<vertex_descriptor_t> augmenting_path_w = edmonds_find_augmenting_path(w, root[w], parent); // Finds path from w to root[w]

	augmenting_path.assign( augmenting_path_v.rbegin(), augmenting_path_v.rend() );
	augmenting_path.insert( augmenting_path.end(), augmenting_path_w.begin(), augmenting_path_w.end() );

	print_augmenting_path("Augmenting path P (before expanding blossoms)", augmenting_path, G_prime_with_contractions, mates);

	// If there's any blossoms in the path, expand
	while(true){
		vector<vertex_descriptor_t>::iterator it;
		set<vertex_descriptor_t> blossom_adjacent_vertices;
		for(it = augmenting_path.begin(); it < augmenting_path.end(); ++it){
			if( !blossom[*it].empty() ){
				
				continue;
			}
		}
		break;
	}

	print_augmenting_path("Augmenting path P", augmenting_path); 
	return augmenting_path;

	//vector<vertex_descriptor_t> augmenting_path;

	/////////////////////////////////////////////////////////////////////////////////////
	////
	//// Initial data structures to keep track of information for implicit matching
	////
	//map< vertex_descriptor_t, vector<vertex_descriptor_t> > blossoms;
	//vector<vertex_descriptor_t> parent( num_vertices(G_prime), graph_traits<Graph>::null_vertex() );
	//vector<vertex_descriptor_t> root( num_vertices(G_prime), graph_traits<Graph>::null_vertex() );
	//vector<size_t> level( num_vertices(G_prime) );	// Level of vertex in F
	//map<edge_descriptor_t, bool> marked_edges;
	//map<vertex_descriptor_t, bool> marked_vertices;
	//vector<vertex_descriptor_t> F;

	//// Mark any matched edges
	//edge_iterator_t ei, ei_end;
	//for( tie(ei, ei_end) = edges(G_prime); ei != ei_end; ++ei){
	//	vertex_descriptor_t v = source( *ei, G_prime );
	//	vertex_descriptor_t w = target( *ei, G_prime );
	//	if( mates[v].find(w) != mates[v].end() ){
	//		marked_edges[*ei] = true;
	//		edge_descriptor_t reverse_current_edge = edge( target(*ei, G_prime), source(*ei, G_prime), G_prime ).first;
	//		bool reverse_current_edge_exists = edge( target(*ei, G_prime), source(*ei, G_prime), G_prime ).second;
	//		if( reverse_current_edge_exists ){
	//			marked_edges[reverse_current_edge] = true;	// Make sure reverse ordering of edge is marked as true
	//		}
	//	}
	//	else{
	//		marked_edges[*ei] = false;
	//	}
	//}

	//// Add exposed vertices to F
	//vertex_iterator_t vi, vi_end;
	//size_t total_num_exposed_subvertices = 0;	// Keep track of the total number of exposed subvertices, if only 1, exit
	//for(boost::tie(vi,vi_end) = vertices(G_prime); vi != vi_end; ++vi){
	//	vertex_descriptor_t v = *vi;
	//	if( num_exposed_subvertices[v] >= 1 ){
	//		total_num_exposed_subvertices += num_exposed_subvertices[v];
	//		// Set the parent, root and level of the the vertex v (where (v,w) is the current edge), where it is itself
	//		parent[v] = v;
	//		root[v] = v;
	//		level[v] = 0;

	//		// Set v as unmarked and add to F
	//		marked_vertices[v] = false;
	//		F.push_back(v);
	//	}
	//}

	//cout_and_log_file << "\tedmonds_implicit_matching:  Total number of exposed subvertices: " << total_num_exposed_subvertices << endl;
	//if( total_num_exposed_subvertices <= 1 ){
	//	return augmenting_path;
	//}

	//vector<vertex_descriptor_t>::iterator it;
	//bool augmenting_path_found = false;
	//while( !F.empty() && !augmenting_path_found ){
	//	
	//	vertex_descriptor_t v = F.back();
	//	F.pop_back();

	//	// Check if vertex is marked, skip
	//	if( marked_vertices[v] ){
	//		continue;
	//	}

	//	// If vertex is odd, mark and skip
	//	if( level[v] % 2 == 1 ){
	//		marked_vertices[v] = true;
	//		continue;
	//	}

	//	out_edge_iterator_t ei, ei_end;
	//	for(tie(ei,ei_end) = out_edges(v, G_prime); ei != ei_end; ++ei){
	//		// Current edge info
	//		edge_descriptor_t current_edge = *ei;
	//		vertex_descriptor_t w = target(current_edge, G_prime);

	//		// Check if edge is marked, skip
	//		if( marked_edges[current_edge] ){
	//			continue;
	//		}

	//		// Check if reverse edge is marked ( (w,v) vs. (v,w) )
	//		edge_descriptor_t reverse_current_edge = edge( target(*ei, G_prime), source(*ei, G_prime), G_prime ).first;
	//		bool reverse_current_edge_exists = edge( target(*ei, G_prime), source(*ei, G_prime), G_prime ).second;
	//		if( reverse_current_edge_exists && marked_edges[reverse_current_edge] ){
	//			continue;
	//		}

	//		///////////////////////////////////////////////////////////////////////////////////
	//		//
	//		// Cases of Edmond's algorithm (implicit), while there exists even unmarked edges
	//		//

	//		// Case 1: If w is not in F (if in F, root and parent are set)
	//		if( root[w] == graph_traits<Graph>::null_vertex() ){
	//			// Mark the parent, root and level of the the vertex w
	//			parent[w] = v;
	//			root[w] = root[v];
	//			level[w] = level[v]+1;

	//			// Set w as unmarked and add to F
	//			marked_vertices[w] = false;
	//			F.push_back(w);

	//			set<vertex_descriptor_t> x_all = mates[w];	// As it is implicit representation, w can have 1 or more mates
	//			set<vertex_descriptor_t>::iterator sit;
	//			for(sit = x_all.begin(); sit != x_all.end(); ++sit){
	//				vertex_descriptor_t x = *sit;

	//				// x could already be in F, so check that it is not
	//				if( root[x] == graph_traits<Graph>::null_vertex() ){ // True if x not in F
	//					// Mark the parent, root and level of the the vertex x (where (w,x) is the matched edge being added to F)
	//					parent[x] = w;
	//					root[x] = root[w];
	//					level[x] = level[w]+1;

	//					// Set x as unmarked and add to F
	//					marked_vertices[x] = false;
	//					F.push_back(x);
	//				}
	//				else{ // x is in F
	//					continue;
	//				}
	//			}
	//		}

	//		// Case 2: w is a blossom, which is already in F (there exists an augmenting path)
	//		else if( !blossoms[w].empty() ){

	//			augmenting_path_found = true;
	//			break;
	//		}

	//		// Case 3: Root of v is different that root of w, we have an augmenting path from exposed vertex to exposed vertex
	//		else if( root[v] != root[w] ){
	//			vector<vertex_descriptor_t> augmenting_path_v = edmonds_find_augmenting_path(v, root[v], parent); // Finds path from v to root[v] (must insert reverse of this into augmenting_path)							
	//			vector<vertex_descriptor_t> augmenting_path_w = edmonds_find_augmenting_path(w, root[w], parent); // Finds path from w to root[w]

	//			augmenting_path.insert( augmenting_path.end(), augmenting_path_v.rbegin(), augmenting_path_v.rend() );
	//			augmenting_path.insert( augmenting_path.end(), augmenting_path_w.begin(), augmenting_path_w.end() );

	//			//print_augmenting_path("Augmenting path P (Case 3)", augmenting_path, G_prime, mates);

	//			augmenting_path_found = true;
	//			break;
	//		}
	//		// Case 4: Found blossom
	//		else{	
	//			// Find common ancestor of v and w, which is where augmenting_path_v and augmenting_path_w both have same vertex
	//			DEBUG_START_L2("Find blossom, contract and lift P' to P ... ");
	//			vector<vertex_descriptor_t> augmenting_path_v = edmonds_find_augmenting_path(v, root[v], parent); // Finds path from v to root[v]
	//			vector<vertex_descriptor_t> augmenting_path_w = edmonds_find_augmenting_path(w, root[w], parent); // Finds path from w to root[w]
	//			
	//			log_file << "\t(v,w) = (" << v << "," << w << ")." << endl;
	//			//print_augmenting_path("augmenting_path_v", augmenting_path_v, G_prime, mates, false);
	//			//print_augmenting_path("augmenting_path_w", augmenting_path_w, G_prime, mates, false);

	//			ptrdiff_t v_w_size_difference = augmenting_path_v.size() - augmenting_path_w.size();
	//			if( v_w_size_difference > 0 ){
	//				augmenting_path_v.erase( augmenting_path_v.begin(), augmenting_path_v.begin() + v_w_size_difference );
	//			}
	//			else if( v_w_size_difference < 0 ){
	//				augmenting_path_w.erase( augmenting_path_w.begin(), augmenting_path_w.begin() - v_w_size_difference );
	//			}

	//			if( v_w_size_difference != 0 ){
	//				//print_augmenting_path("augmenting_path_v (after making same size)", augmenting_path_v, G_prime, mates, false);
	//				//print_augmenting_path("augmenting_path_w (after making same size)", augmenting_path_w, G_prime, mates, false);
	//			}

	//			while( augmenting_path_v.front() != augmenting_path_w.front() && !augmenting_path_v.empty() ){
	//				augmenting_path_v.erase(augmenting_path_v.begin());
	//				augmenting_path_w.erase(augmenting_path_w.begin());
	//			}

	//			//print_augmenting_path("augmenting_path_v (after erasing up to start of blossom)", augmenting_path_v, G_prime, mates, false);
	//			//print_augmenting_path("augmenting_path_w (after erasing up to start of blossom)", augmenting_path_w, G_prime, mates, false);

	//			if( augmenting_path_v.empty() || augmenting_path_w.empty() ){
	//				cout_and_log_file << "ERROR! No common ancestor found for blossom." << endl;
	//				exit_program();
	//			}
	//			vertex_descriptor_t start_blossom = augmenting_path_v.front();
	//			log_file << "Start of blossom: " << start_blossom << endl;

	//			// Case 4a: Blossom, but the start of the blossom has more than 2 exposed subvertices, which is an augmenting path, 
	//			//				so don't bother contracting (start of blossom must have more than two exposed subvertices, as performing matching augmentation adds two matching edges)
	//			if( num_exposed_subvertices[ start_blossom ] >= 2 ){
	//				vector<vertex_descriptor_t> augmenting_path_v = edmonds_find_augmenting_path(v, start_blossom, parent); // Finds path from v to start_blossom (must insert reverse of this into augmenting_path)							
	//				vector<vertex_descriptor_t> augmenting_path_w = edmonds_find_augmenting_path(w, start_blossom, parent); // Finds path from w to start_blossom

	//				augmenting_path.insert( augmenting_path.end(), augmenting_path_v.rbegin(), augmenting_path_v.rend() );
	//				augmenting_path.insert( augmenting_path.end(), augmenting_path_w.begin(), augmenting_path_w.end() );

	//				//print_augmenting_path("Augmenting path P (Case 4a: Start of blossom has 2 or more exposed subvertices)", augmenting_path, G_prime, mates);
	//				
	//				augmenting_path_found = true;
	//				break;
	//			}

	//			// Case 4b: Blossom with either 0 or 1 exposed subvertices for the start of the blossom

	//			//DEBUG_END_L2("Find start of blossom ... ");

	//			// Find blossom
	//			//DEBUG_START_L2("Find blossom ... ");
	//			vector<vertex_descriptor_t> blossom;	// Start vertex of blossom listed at beginning and end
	//			vector<vertex_descriptor_t> augmenting_path_v_start = edmonds_find_augmenting_path(v, start_blossom, parent); // Finds path from v to start_blossom
	//			vector<vertex_descriptor_t> augmenting_path_w_start = edmonds_find_augmenting_path(w, start_blossom, parent); // Finds path from w to start_blossom
	//			blossom.insert( blossom.end(), augmenting_path_v_start.rbegin(), augmenting_path_v_start.rend() );
	//			blossom.insert( blossom.end(), augmenting_path_w_start.begin(), augmenting_path_w_start.end() );
	//			//print_augmenting_path("Blossom", blossom, G_prime, mates);
	//			//DEBUG_END_L2("Find blossom ... ");

	//			//DEBUG_START_L2("Contract blossom ... ");
	//			vertex_descriptor_t blossom_vertex = add_vertex(G_prime);
	//			log_file << "Contracted blossom vertex number: " << blossom_vertex << endl;

	//			blossoms[blossom_vertex] = blossom;

	//			vector<vertex_descriptor_t>::iterator it;
	//			set<vertex_descriptor_t> blossom_adjacent_vertices;
	//			vector<vertex_pair_t> prev_blossom_edges;
	//			for(it = blossom.begin(); it < blossom.end()-1; ++it){	// Don't check start_blossom twice (blossom.end()-1)
	//				vertex_descriptor_t blossom_u = *it;
	//				out_edge_iterator_t ei, ei_end;
	//				for(boost::tie(ei,ei_end) = out_edges(blossom_u, G_prime); ei != ei_end; ++ei){
	//					vertex_descriptor_t blossom_v = target(*ei, G_prime);
	//					if( blossom_u < blossom_v ){
	//						prev_blossom_edges.push_back( make_pair(blossom_u, blossom_v) );
	//					}
	//					else{
	//						prev_blossom_edges.push_back( make_pair(blossom_v, blossom_u) );
	//					}

	//					// If blossom vertex was a mate to a vertex in G', set it to be a mate with the contracted blossom vertex
	//					if( mates[blossom_u].find(blossom_v) != mates[blossom_u].end() ){
	//						mates[blossom_v].insert(blossom_vertex);
	//						mates[blossom_vertex].insert(blossom_v);
	//					}
	//					blossom_adjacent_vertices.insert( blossom_v );	// Does not insert duplicates
	//				}
	//			}

	//			set<vertex_descriptor_t>::iterator sit;
	//			for(sit = blossom_adjacent_vertices.begin(); sit != blossom_adjacent_vertices.end(); ++sit){
	//				add_edge(blossom_vertex, *sit, G_prime);
	//			}

	//			// Remove all edges to vertices in blossom (but remember for after exiting recursion), so they appear to not exist in matching algorithm
	//			for(it = blossom.begin(); it < blossom.end()-1; ++it){
	//				clear_vertex(*it, G_prime);
	//			}

	//			// If the root (start_blossom) of the blossom was exposed (failed cased 4a due to having only 1 exposed subvertex), the blossom has 1 exposed subvertex, 0 otherwise
	//			num_exposed_subvertices[blossom_vertex] = num_exposed_subvertices[start_blossom];
	//			//root.push_back( root[start_blossom] );
	//			//parent.push_back( parent[start_blossom] );
	//			//level.push_back( level[start_blossom] );
	//			log_file << "Contracted blossom vertex number of exposed subvertices: " << num_exposed_subvertices[blossom_vertex] << endl;

	//			//for(it = blossom.begin(); it < blossom.end()-1; ++it){
	//			//	// Remove any edges in even_edges which have blossom vertices in them, replace with the contracted blossom vertex
	//			//	//replace(even_edges_source.begin(), even_edges_source.end(), *it, blossom_vertex);
	//			//	//replace(even_edges_target.begin(), even_edges_target.end(), *it, blossom_vertex);
	//			//}
	//			//DEBUG_END_L2("Contract blossom ... ");

	//			// Find P'
	//			//DEBUG_RECURSION_START("Find P' (recursive call on edmonds_implicit_matching with contracted graph) ... ");
	//			//augmenting_path = edmonds_implicit_matching(G_prime, mates, num_exposed_subvertices);
	//			//print_augmenting_path("Augmenting path P'", augmenting_path, G_prime, mates);
	//			//DEBUG_RECURSION_END("Find P' (recursive call on edmonds_implicit_matching with contracted graph) ... ");

	//			// Restore vertices/edges
	//			clear_vertex(blossom_vertex, G_prime);
	//			remove_vertex(blossom_vertex, G_prime);
	//			vector<vertex_pair_t>::iterator vvit;
	//			for(vvit = prev_blossom_edges.begin(); vvit < prev_blossom_edges.end(); ++vvit){
	//				add_edge( (*vvit).first, (*vvit).second, G_prime );
	//			}
	//			
	//			// Find P using P'
	//			//DEBUG_START_L2("Find P using P'");
	//		//	if( !augmenting_path.empty() ){

	//		//		vector<vertex_descriptor_t> augmenting_path_before_blossom;
	//		//		vector<vertex_descriptor_t> augmenting_path_blossom_at_start;
	//		//		vector<vertex_descriptor_t> augmenting_path_blossom_at_middle;
	//		//		vector<vertex_descriptor_t> augmenting_path_in_between_blossoms;
	//		//		vector<vertex_descriptor_t> augmenting_path_blossom_at_end;
	//		//		vector<vertex_descriptor_t> augmenting_path_after_blossom;

	//		//		vector<vertex_descriptor_t>::iterator in_P_before_blossom = augmenting_path.end();
	//		//		vector<vertex_descriptor_t>::iterator in_P_after_blossom = augmenting_path.end();
	//		//		vector<vertex_descriptor_t>::iterator blossom_it, blossom_back_it;

	//		//		// Blossom at start of path
	//		//		if( blossom_vertex == augmenting_path.front() ){
	//		//			in_P_after_blossom = augmenting_path.begin()+1; // returns 2nd element in augmenting_path
	//		//			log_file << "Vertex in P' after blossom: " << *in_P_after_blossom << endl;
	//		//			// Have two iterators, one start at front and check every second one, one at back checking every second up to the front (neither checking the same vertices)
	//		//			for(blossom_it = blossom.begin(), blossom_back_it = blossom.end() - 1; blossom_it < blossom.end() -1 || blossom_back_it > blossom.begin(); ){
	//		//				if( edge(*blossom_it, *in_P_after_blossom, G_prime).second && mates[*blossom_it].find(*in_P_after_blossom) == mates[*blossom_it].end()  ){	// If there's an edge from blossom_it to P' and its not a matching
	//		//					log_file << "Vertex in blossom connected to vertex in P': " << *blossom_it << endl;							
	//		//					augmenting_path_blossom_at_start.assign( blossom.begin(), blossom_it+1 );
	//		//					break;
	//		//				}
	//		//				else if( edge(*blossom_back_it, *in_P_after_blossom, G_prime).second && mates[*blossom_back_it].find(*in_P_after_blossom) == mates[*blossom_back_it].end()  ){	// If there's an edge from blossom_back_it to P' and its not a matching
	//		//					log_file << "Vertex in blossom (back iterator) connected to vertex in P': " << *blossom_back_it << endl;							
	//		//					augmenting_path_blossom_at_start.assign( blossom_back_it, blossom.end() );
	//		//					reverse( augmenting_path_blossom_at_start.begin(), augmenting_path_blossom_at_start.end() );
	//		//					break;

	//		//				}

	//		//				// Increment iterators
	//		//				if( blossom_it < blossom.end()-1 ){
	//		//					blossom_it += 2;
	//		//				}
	//		//				if( blossom_back_it > blossom.begin() ){
	//		//					blossom_back_it -= 2;
	//		//				}
	//		//			}
	//		//			augmenting_path_after_blossom.assign( in_P_after_blossom, augmenting_path.end() ); // Fill in vertices after the blossom vertex in P'
	//		//		}
	//		//		// Blossom at end of path
	//		//		else if( blossom_vertex == augmenting_path.back() ){
	//		//			in_P_before_blossom = augmenting_path.end()-2; // returns 2nd to last element in augmenting_path
	//		//			log_file << "Vertex in P' before blossom: " << *in_P_before_blossom << endl;

	//		//			// Have two iterators, one start at front and check every second one, one at back checking every second up to the front (neither checking the same vertices)
	//		//			for(blossom_it = blossom.begin(), blossom_back_it = blossom.end() - 1; blossom_it < blossom.end() -1 || blossom_back_it > blossom.begin(); ){
	//		//				if( edge(*blossom_it, *in_P_before_blossom, G_prime).second && mates[*blossom_it].find(*in_P_before_blossom) == mates[*blossom_it].end()  ){	// If there's an edge from P' to blossom_it and its not a matching
	//		//					log_file << "Vertex in blossom connected from vertex in P': " << *blossom_it << endl;
	//		//					augmenting_path_blossom_at_end.assign( blossom.begin(), blossom_it+1 );
	//		//					reverse( augmenting_path_blossom_at_end.begin(), augmenting_path_blossom_at_end.end() );
	//		//					break;
	//		//				}
	//		//				else if( edge(*blossom_back_it, *in_P_before_blossom, G_prime).second && mates[*blossom_back_it].find(*in_P_before_blossom) == mates[*blossom_back_it].end()  ){	// If there's an edge from P' to blossom_back_it and its not a matching
	//		//					log_file << "Vertex in blossom (back iterator) connected from vertex in P': " << *blossom_back_it << endl;
	//		//					augmenting_path_blossom_at_end.assign( blossom_back_it, blossom.end() );
	//		//					break;
	//		//				}

	//		//				// Increment iterators
	//		//				if( blossom_it < blossom.end()-1 ){
	//		//					blossom_it += 2;
	//		//				}
	//		//				if( blossom_back_it > blossom.begin() ){
	//		//					blossom_back_it -= 2;
	//		//				}
	//		//			}
	//		//			augmenting_path_before_blossom.assign( augmenting_path.begin(), in_P_before_blossom+1 ); // Fill in vertices before the blossom vertex in P'
	//		//		}
	//		//		// Blossom possibly in middle of path
	//		//		else{
	//		//			// Find path, not including blossom
	//		//			vector<vertex_descriptor_t>::iterator it;
	//		//			size_t blossom_position_in_P = 0;
	//		//			for(it = augmenting_path.begin(); it < augmenting_path.end(); ++it){
	//		//				if( *it == blossom_vertex ){
	//		//					in_P_before_blossom = it-1;
	//		//					in_P_after_blossom = it+1;
	//		//					log_file << "Vertex in P' which connects to blossom: "  << *in_P_before_blossom << endl;
	//		//					log_file << "Vertex in P' which connects from blossom: " << *in_P_after_blossom << endl;

	//		//					// Fill in vertices before the blossom vertex in P' and after
	//		//					augmenting_path_before_blossom.assign( augmenting_path.begin(), in_P_before_blossom+1 );
	//		//					augmenting_path_after_blossom.assign( in_P_after_blossom, augmenting_path.end() );

	//		//					// If the P' connects (and is matching) to the start of the blossom, just add the start vertex to the path
	//		//					if( blossom_position_in_P % 2 == 0 && mates[*in_P_before_blossom].find(blossom.front()) != mates[*in_P_before_blossom].end() && mates[*in_P_after_blossom].find(blossom.front()) == mates[*in_P_after_blossom].end() && edge(*in_P_after_blossom, blossom.front(), G_prime).second ){ // If even, blossom has matched edge into blossom, unmatched out (must check unmatched edge exists)
	//		//						log_file << "P' connects to blossom start (matched in, unmatched out): " << blossom.front() << endl;
	//		//						augmenting_path_blossom_at_middle.push_back( blossom.front() );
	//		//						break;
	//		//					}
	//		//					else if( blossom_position_in_P % 2 == 1 && mates[*in_P_before_blossom].find(blossom.front()) == mates[*in_P_before_blossom].end() && mates[*in_P_after_blossom].find(blossom.front()) != mates[*in_P_after_blossom].end() && edge(*in_P_before_blossom, blossom.front(), G_prime).second ){ // If odd, blossom has unmatched edge into blossom, matched out (must check unmatched edge exists)
	//		//						log_file << "P' connects to blossom start (unmatched in, matched out): " << blossom.front() << endl;
	//		//						augmenting_path_blossom_at_middle.push_back( blossom.front() );
	//		//						break;
	//		//					}

	//		//					// Look for unmatched edge in/out of blossom
	//		//					// Have two iterators, one start at front and check every second one, one at back checking every second up to the front (neither checking the same vertices)
	//		//					for(blossom_it = blossom.begin(), blossom_back_it = blossom.end() - 1; blossom_it < blossom.end() -1 || blossom_back_it > blossom.begin(); ){
	//		//						if( blossom_position_in_P % 2 == 0 ){ // If even, blossom has matched edge into blossom, unmatched out (opposite if false)
	//		//							if( edge(*blossom_it, *in_P_after_blossom, G_prime).second && mates[*blossom_it].find(*in_P_after_blossom) == mates[*blossom_it].end()  ){	// True if there's an edge exiting the blossom and its not a matching
	//		//								log_file << "Vertex in blossom connected to vertex in P': " << *blossom_it << endl;		
	//		//								if( mates[*blossom_it].find(*in_P_before_blossom) != mates[*blossom_it].end() ){
	//		//									log_file << "Matched edge from in_P_before_blossom to start_blossom exists." << endl;
	//		//								}
	//		//								
	//		//								augmenting_path_blossom_at_middle.assign( blossom.begin(), blossom_it+1 );
	//		//								break;
	//		//							}
	//		//							else if( edge(*blossom_back_it, *in_P_after_blossom, G_prime).second && mates[*blossom_back_it].find(*in_P_after_blossom) == mates[*blossom_back_it].end()  ){	// True if there's an edge exiting the blossom and its not a matching
	//		//								log_file << "Vertex in blossom (back iterator) connected to vertex in P': " << *blossom_back_it << endl;		
	//		//								if( mates[*blossom_back_it].find(*in_P_before_blossom) != mates[*blossom_back_it].end() ){
	//		//									log_file << "Matched edge from in_P_before_blossom to start_blossom exists." << endl;
	//		//								}

	//		//								augmenting_path_blossom_at_middle.assign( blossom_back_it, blossom.end() );
	//		//								reverse( augmenting_path_blossom_at_middle.begin(), augmenting_path_blossom_at_middle.end() );
	//		//								break;
	//		//							}
	//		//						}
	//		//						else{
	//		//							if( edge(*blossom_it, *in_P_before_blossom, G_prime).second && mates[*blossom_it].find(*in_P_before_blossom) == mates[*blossom_it].end()  ){	// True if there's an edge into the blossom and its not a matching
	//		//								log_file << "Vertex in blossom connected from vertex in P': " << *blossom_it << endl;										
	//		//								if( mates[*blossom_it].find(*in_P_after_blossom) != mates[*blossom_it].end() ){
	//		//									log_file << "Matched edge from in_P_after_blossom to start_blossom exists." << endl;
	//		//								}

	//		//								augmenting_path_blossom_at_middle.assign( blossom.begin(), blossom_it+1 );
	//		//								reverse( augmenting_path_blossom_at_middle.begin(), augmenting_path_blossom_at_middle.end() );
	//		//								break;
	//		//							}
	//		//							else if( edge(*blossom_back_it, *in_P_before_blossom, G_prime).second && mates[*blossom_back_it].find(*in_P_before_blossom) == mates[*blossom_back_it].end()  ){	// True if there's an edge into the blossom and its not a matching
	//		//								log_file << "Vertex in blossom (back iterator) connected to vertex in P': " << *blossom_back_it << endl;		
	//		//								if( mates[*blossom_back_it].find(*in_P_after_blossom) != mates[*blossom_back_it].end() ){
	//		//									log_file << "Matched edge from in_P_after_blossom to start_blossom exists." << endl;
	//		//								}

	//		//								augmenting_path_blossom_at_middle.assign( blossom_it, blossom.end() );
	//		//								break;
	//		//							}
	//		//						}

	//		//						// Increment iterators
	//		//						if( blossom_it < blossom.end()-1 ){
	//		//							blossom_it += 2;
	//		//						}
	//		//						if( blossom_back_it > blossom.begin() ){
	//		//							blossom_back_it -= 2;
	//		//						}
	//		//					}
	//		//					//log_file << "ERROR!!! Went through blossom and did not find blossom vertex connected to P'" << endl;
	//		//					break;
	//		//				}
	//		//				blossom_position_in_P++;
	//		//			}
	//		//		}

	//		//		if( in_P_before_blossom != augmenting_path.end() || in_P_after_blossom != augmenting_path.end() ){ // True if there's a bossom in P'
	//		//			augmenting_path.clear();

	//		//			if( !augmenting_path_before_blossom.empty() ){
	//		//				augmenting_path.assign( augmenting_path_before_blossom.begin(), augmenting_path_before_blossom.end() );
	//		//				//print_augmenting_path("augmenting_path_before_blossom", augmenting_path_before_blossom, G_prime, mates, false);
	//		//			}
	//		//			if( !augmenting_path_blossom_at_start.empty() ){
	//		//				augmenting_path.insert( augmenting_path.end(), augmenting_path_blossom_at_start.begin(), augmenting_path_blossom_at_start.end() );
	//		//				//print_augmenting_path("augmenting_path_blossom_at_start", augmenting_path_blossom_at_start, G_prime, mates, false);
	//		//			}
	//		//			if( !augmenting_path_blossom_at_middle.empty() ){
	//		//				augmenting_path.insert( augmenting_path.end(), augmenting_path_blossom_at_middle.begin(), augmenting_path_blossom_at_middle.end() );
	//		//				//print_augmenting_path("augmenting_path_blossom_at_middle", augmenting_path_blossom_at_middle, G_prime, mates, false);
	//		//			}
	//		//			if( !augmenting_path_in_between_blossoms.empty() ){
	//		//				augmenting_path.insert( augmenting_path.end(), augmenting_path_in_between_blossoms.begin(), augmenting_path_in_between_blossoms.end() );
	//		//				//print_augmenting_path("augmenting_path_in_between_blossoms", augmenting_path_in_between_blossoms, G_prime, mates, false);
	//		//			}
	//		//			if( !augmenting_path_blossom_at_end.empty() ){
	//		//				augmenting_path.insert( augmenting_path.end(), augmenting_path_blossom_at_end.begin(), augmenting_path_blossom_at_end.end() );
	//		//				//print_augmenting_path("augmenting_path_blossom_at_end", augmenting_path_blossom_at_end, G_prime, mates, false);
	//		//			}
	//		//			if( !augmenting_path_after_blossom.empty() ){
	//		//				augmenting_path.insert( augmenting_path.end(), augmenting_path_after_blossom.begin(), augmenting_path_after_blossom.end() );
	//		//				//print_augmenting_path("augmenting_path_after_blossom", augmenting_path_after_blossom, G_prime, mates, false);
	//		//			}
	//		//		}

	//		//		set<vertex_pair_t> augmenting_path_edges;	// Use to check that no edge is inserted twice in a path
	//		//		vector<vertex_descriptor_t>::iterator it;
	//		//		bool not_augmenting_path = false;
	//		//		for(it = augmenting_path.begin(); it < augmenting_path.end()-1; ++it){
	//		//			pair<set<vertex_pair_t>::iterator, bool> ret;
	//		//			if( it[0] < it[1] ){
	//		//				ret = augmenting_path_edges.insert( make_pair( it[0], it[1] ) );
	//		//			}
	//		//			else{
	//		//				ret = augmenting_path_edges.insert( make_pair( it[1], it[0] ) );
	//		//			}

	//		//			if( ret.second == false ){ // True if value not inserted
	//		//				log_file << "Returned augmenting path (from recursive call) contained same edge twice." << endl;
	//		//				not_augmenting_path = true;
	//		//				break;
	//		//			}
	//		//		}

	//		//		if( not_augmenting_path ){
	//		//			marked_edges[current_edge] = true;
	//		//			if( reverse_current_edge_exists){
	//		//				marked_edges[reverse_current_edge] = true;	// Make sure reverse ordering of edge is marked as true
	//		//			}
	//		//			continue;
	//		//		}
	//		//	}
	//		//	
	//		//	//print_augmenting_path("Augmenting path P", augmenting_path, G_prime, mates);
	//		//	//DEBUG_END_L2("Find P using P'");

	//		//	DEBUG_END_L2("Find blossom, contract and lift P' to P ... ");
	//		//	return augmenting_path;
	//		//}

	//			marked_edges[current_edge] = true;
	//			if( reverse_current_edge_exists){
	//				marked_edges[reverse_current_edge] = true;	// Make sure reverse ordering of edge is marked as true
	//			}
	//		}
	//		marked_vertices[v] = true;
	//	}
	//}

	//// Didn't find path, so return empty path
	//if( augmenting_path.empty() ){
	//	return augmenting_path;
	//}

	////// Found augmenting path, uncontract all contracted blossoms, then return augmenting path
	////vector<vertex_descriptor_t> augmenting_path_before_blossom;
	////vector<vertex_descriptor_t> augmenting_path_blossom_at_start;
	////vector<vertex_descriptor_t> augmenting_path_blossom_at_middle;
	////vector<vertex_descriptor_t> augmenting_path_in_between_blossoms;
	////vector<vertex_descriptor_t> augmenting_path_blossom_at_end;
	////vector<vertex_descriptor_t> augmenting_path_after_blossom;

	////vector<vertex_descriptor_t>::iterator in_P_before_blossom = augmenting_path.end();
	////vector<vertex_descriptor_t>::iterator in_P_after_blossom = augmenting_path.end();
	////vector<vertex_descriptor_t>::iterator blossom_it, blossom_back_it;

	////// Blossom at start of path
	////if( blossom_vertex == augmenting_path.front() ){
	////	in_P_after_blossom = augmenting_path.begin()+1; // returns 2nd element in augmenting_path
	////	log_file << "Vertex in P' after blossom: " << *in_P_after_blossom << endl;
	////	// Have two iterators, one start at front and check every second one, one at back checking every second up to the front (neither checking the same vertices)
	////	for(blossom_it = blossom.begin(), blossom_back_it = blossom.end() - 1; blossom_it < blossom.end() -1 || blossom_back_it > blossom.begin(); ){
	////		if( edge(*blossom_it, *in_P_after_blossom, G_prime).second && mates[*blossom_it].find(*in_P_after_blossom) == mates[*blossom_it].end()  ){	// If there's an edge from blossom_it to P' and its not a matching
	////			log_file << "Vertex in blossom connected to vertex in P': " << *blossom_it << endl;							
	////			augmenting_path_blossom_at_start.assign( blossom.begin(), blossom_it+1 );
	////			break;
	////		}
	////		else if( edge(*blossom_back_it, *in_P_after_blossom, G_prime).second && mates[*blossom_back_it].find(*in_P_after_blossom) == mates[*blossom_back_it].end()  ){	// If there's an edge from blossom_back_it to P' and its not a matching
	////			log_file << "Vertex in blossom (back iterator) connected to vertex in P': " << *blossom_back_it << endl;							
	////			augmenting_path_blossom_at_start.assign( blossom_back_it, blossom.end() );
	////			reverse( augmenting_path_blossom_at_start.begin(), augmenting_path_blossom_at_start.end() );
	////			break;

	////		}

	////		// Increment iterators
	////		if( blossom_it < blossom.end()-1 ){
	////			blossom_it += 2;
	////		}
	////		if( blossom_back_it > blossom.begin() ){
	////			blossom_back_it -= 2;
	////		}
	////	}
	////	augmenting_path_after_blossom.assign( in_P_after_blossom, augmenting_path.end() ); // Fill in vertices after the blossom vertex in P'
	////}
	////// Blossom at end of path
	////else if( blossom_vertex == augmenting_path.back() ){
	////	in_P_before_blossom = augmenting_path.end()-2; // returns 2nd to last element in augmenting_path
	////	log_file << "Vertex in P' before blossom: " << *in_P_before_blossom << endl;

	////	// Have two iterators, one start at front and check every second one, one at back checking every second up to the front (neither checking the same vertices)
	////	for(blossom_it = blossom.begin(), blossom_back_it = blossom.end() - 1; blossom_it < blossom.end() -1 || blossom_back_it > blossom.begin(); ){
	////		if( edge(*blossom_it, *in_P_before_blossom, G_prime).second && mates[*blossom_it].find(*in_P_before_blossom) == mates[*blossom_it].end()  ){	// If there's an edge from P' to blossom_it and its not a matching
	////			log_file << "Vertex in blossom connected from vertex in P': " << *blossom_it << endl;
	////			augmenting_path_blossom_at_end.assign( blossom.begin(), blossom_it+1 );
	////			reverse( augmenting_path_blossom_at_end.begin(), augmenting_path_blossom_at_end.end() );
	////			break;
	////		}
	////		else if( edge(*blossom_back_it, *in_P_before_blossom, G_prime).second && mates[*blossom_back_it].find(*in_P_before_blossom) == mates[*blossom_back_it].end()  ){	// If there's an edge from P' to blossom_back_it and its not a matching
	////			log_file << "Vertex in blossom (back iterator) connected from vertex in P': " << *blossom_back_it << endl;
	////			augmenting_path_blossom_at_end.assign( blossom_back_it, blossom.end() );
	////			break;
	////		}

	////		// Increment iterators
	////		if( blossom_it < blossom.end()-1 ){
	////			blossom_it += 2;
	////		}
	////		if( blossom_back_it > blossom.begin() ){
	////			blossom_back_it -= 2;
	////		}
	////	}
	////	augmenting_path_before_blossom.assign( augmenting_path.begin(), in_P_before_blossom+1 ); // Fill in vertices before the blossom vertex in P'
	////}
	////// Blossom possibly in middle of path
	////else{
	////	// Find path, not including blossom
	////	vector<vertex_descriptor_t>::iterator it;
	////	size_t blossom_position_in_P = 0;
	////	for(it = augmenting_path.begin(); it < augmenting_path.end(); ++it){
	////		if( *it == blossom_vertex ){
	////			in_P_before_blossom = it-1;
	////			in_P_after_blossom = it+1;
	////			log_file << "Vertex in P' which connects to blossom: "  << *in_P_before_blossom << endl;
	////			log_file << "Vertex in P' which connects from blossom: " << *in_P_after_blossom << endl;

	////			// Fill in vertices before the blossom vertex in P' and after
	////			augmenting_path_before_blossom.assign( augmenting_path.begin(), in_P_before_blossom+1 );
	////			augmenting_path_after_blossom.assign( in_P_after_blossom, augmenting_path.end() );

	////			// If the P' connects (and is matching) to the start of the blossom, just add the start vertex to the path
	////			if( blossom_position_in_P % 2 == 0 && mates[*in_P_before_blossom].find(blossom.front()) != mates[*in_P_before_blossom].end() && mates[*in_P_after_blossom].find(blossom.front()) == mates[*in_P_after_blossom].end() && edge(*in_P_after_blossom, blossom.front(), G_prime).second ){ // If even, blossom has matched edge into blossom, unmatched out (must check unmatched edge exists)
	////				log_file << "P' connects to blossom start (matched in, unmatched out): " << blossom.front() << endl;
	////				augmenting_path_blossom_at_middle.push_back( blossom.front() );
	////				break;
	////			}
	////			else if( blossom_position_in_P % 2 == 1 && mates[*in_P_before_blossom].find(blossom.front()) == mates[*in_P_before_blossom].end() && mates[*in_P_after_blossom].find(blossom.front()) != mates[*in_P_after_blossom].end() && edge(*in_P_before_blossom, blossom.front(), G_prime).second ){ // If odd, blossom has unmatched edge into blossom, matched out (must check unmatched edge exists)
	////				log_file << "P' connects to blossom start (unmatched in, matched out): " << blossom.front() << endl;
	////				augmenting_path_blossom_at_middle.push_back( blossom.front() );
	////				break;
	////			}

	////			// Look for unmatched edge in/out of blossom
	////			// Have two iterators, one start at front and check every second one, one at back checking every second up to the front (neither checking the same vertices)
	////			for(blossom_it = blossom.begin(), blossom_back_it = blossom.end() - 1; blossom_it < blossom.end() -1 || blossom_back_it > blossom.begin(); ){
	////				if( blossom_position_in_P % 2 == 0 ){ // If even, blossom has matched edge into blossom, unmatched out (opposite if false)
	////					if( edge(*blossom_it, *in_P_after_blossom, G_prime).second && mates[*blossom_it].find(*in_P_after_blossom) == mates[*blossom_it].end()  ){	// True if there's an edge exiting the blossom and its not a matching
	////						log_file << "Vertex in blossom connected to vertex in P': " << *blossom_it << endl;		
	////						if( mates[*blossom_it].find(*in_P_before_blossom) != mates[*blossom_it].end() ){
	////							log_file << "Matched edge from in_P_before_blossom to start_blossom exists." << endl;
	////						}
	////						
	////						augmenting_path_blossom_at_middle.assign( blossom.begin(), blossom_it+1 );
	////						break;
	////					}
	////					else if( edge(*blossom_back_it, *in_P_after_blossom, G_prime).second && mates[*blossom_back_it].find(*in_P_after_blossom) == mates[*blossom_back_it].end()  ){	// True if there's an edge exiting the blossom and its not a matching
	////						log_file << "Vertex in blossom (back iterator) connected to vertex in P': " << *blossom_back_it << endl;		
	////						if( mates[*blossom_back_it].find(*in_P_before_blossom) != mates[*blossom_back_it].end() ){
	////							log_file << "Matched edge from in_P_before_blossom to start_blossom exists." << endl;
	////						}

	////						augmenting_path_blossom_at_middle.assign( blossom_back_it, blossom.end() );
	////						reverse( augmenting_path_blossom_at_middle.begin(), augmenting_path_blossom_at_middle.end() );
	////						break;
	////					}
	////				}
	////				else{
	////					if( edge(*blossom_it, *in_P_before_blossom, G_prime).second && mates[*blossom_it].find(*in_P_before_blossom) == mates[*blossom_it].end()  ){	// True if there's an edge into the blossom and its not a matching
	////						log_file << "Vertex in blossom connected from vertex in P': " << *blossom_it << endl;										
	////						if( mates[*blossom_it].find(*in_P_after_blossom) != mates[*blossom_it].end() ){
	////							log_file << "Matched edge from in_P_after_blossom to start_blossom exists." << endl;
	////						}

	////						augmenting_path_blossom_at_middle.assign( blossom.begin(), blossom_it+1 );
	////						reverse( augmenting_path_blossom_at_middle.begin(), augmenting_path_blossom_at_middle.end() );
	////						break;
	////					}
	////					else if( edge(*blossom_back_it, *in_P_before_blossom, G_prime).second && mates[*blossom_back_it].find(*in_P_before_blossom) == mates[*blossom_back_it].end()  ){	// True if there's an edge into the blossom and its not a matching
	////						log_file << "Vertex in blossom (back iterator) connected to vertex in P': " << *blossom_back_it << endl;		
	////						if( mates[*blossom_back_it].find(*in_P_after_blossom) != mates[*blossom_back_it].end() ){
	////							log_file << "Matched edge from in_P_after_blossom to start_blossom exists." << endl;
	////						}

	////						augmenting_path_blossom_at_middle.assign( blossom_it, blossom.end() );
	////						break;
	////					}
	////				}

	////				// Increment iterators
	////				if( blossom_it < blossom.end()-1 ){
	////					blossom_it += 2;
	////				}
	////				if( blossom_back_it > blossom.begin() ){
	////					blossom_back_it -= 2;
	////				}
	////			}
	////			//log_file << "ERROR!!! Went through blossom and did not find blossom vertex connected to P'" << endl;
	////			break;
	////		}
	////		blossom_position_in_P++;
	////	}
	////}

	////if( in_P_before_blossom != augmenting_path.end() || in_P_after_blossom != augmenting_path.end() ){ // True if there's a bossom in P'
	////	augmenting_path.clear();

	////	if( !augmenting_path_before_blossom.empty() ){
	////		augmenting_path.assign( augmenting_path_before_blossom.begin(), augmenting_path_before_blossom.end() );
	////		//print_augmenting_path("augmenting_path_before_blossom", augmenting_path_before_blossom, G_prime, mates, false);
	////	}
	////	if( !augmenting_path_blossom_at_start.empty() ){
	////		augmenting_path.insert( augmenting_path.end(), augmenting_path_blossom_at_start.begin(), augmenting_path_blossom_at_start.end() );
	////		//print_augmenting_path("augmenting_path_blossom_at_start", augmenting_path_blossom_at_start, G_prime, mates, false);
	////	}
	////	if( !augmenting_path_blossom_at_middle.empty() ){
	////		augmenting_path.insert( augmenting_path.end(), augmenting_path_blossom_at_middle.begin(), augmenting_path_blossom_at_middle.end() );
	////		//print_augmenting_path("augmenting_path_blossom_at_middle", augmenting_path_blossom_at_middle, G_prime, mates, false);
	////	}
	////	if( !augmenting_path_in_between_blossoms.empty() ){
	////		augmenting_path.insert( augmenting_path.end(), augmenting_path_in_between_blossoms.begin(), augmenting_path_in_between_blossoms.end() );
	////		//print_augmenting_path("augmenting_path_in_between_blossoms", augmenting_path_in_between_blossoms, G_prime, mates, false);
	////	}
	////	if( !augmenting_path_blossom_at_end.empty() ){
	////		augmenting_path.insert( augmenting_path.end(), augmenting_path_blossom_at_end.begin(), augmenting_path_blossom_at_end.end() );
	////		//print_augmenting_path("augmenting_path_blossom_at_end", augmenting_path_blossom_at_end, G_prime, mates, false);
	////	}
	////	if( !augmenting_path_after_blossom.empty() ){
	////		augmenting_path.insert( augmenting_path.end(), augmenting_path_after_blossom.begin(), augmenting_path_after_blossom.end() );
	////		//print_augmenting_path("augmenting_path_after_blossom", augmenting_path_after_blossom, G_prime, mates, false);
	////	}
	////}

	////set<vertex_pair_t> augmenting_path_edges;	// Use to check that no edge is inserted twice in a path
	////vector<vertex_descriptor_t>::iterator it;
	////bool not_augmenting_path = false;
	////for(it = augmenting_path.begin(); it < augmenting_path.end()-1; ++it){
	////	pair<set<vertex_pair_t>::iterator, bool> ret;
	////	if( it[0] < it[1] ){
	////		ret = augmenting_path_edges.insert( make_pair( it[0], it[1] ) );
	////	}
	////	else{
	////		ret = augmenting_path_edges.insert( make_pair( it[1], it[0] ) );
	////	}

	////	if( ret.second == false ){ // True if value not inserted
	////		log_file << "Returned augmenting path (from recursive call) contained same edge twice." << endl;
	////		not_augmenting_path = true;
	////		break;
	////	}
	////}

	////if( not_augmenting_path ){
	////	marked_edges[current_edge] = true;
	////	if( reverse_current_edge_exists){
	////		marked_edges[reverse_current_edge] = true;	// Make sure reverse ordering of edge is marked as true
	////	}
	////	continue;
	////}
	//
	////print_augmenting_path("Augmenting path P", augmenting_path, G_prime, mates);
	////DEBUG_END_L2("Find P using P'");

	//DEBUG_END_L2("Find blossom, contract and lift P' to P ... ");
	//return augmenting_path;
}

//class greater_than_by_num_exposed_subvertices
//{
//public:
//	greater_than_by_num_exposed_subvertices(map<vertex_descriptor_t, ptrdiff_t> num_exposed_subvertices): m_num_exposed_subvertices(num_exposed_subvertices) {}
//	bool operator() (const vertex_descriptor_t x, const vertex_descriptor_t y)
//	{
//		return m_num_exposed_subvertices[x] > m_num_exposed_subvertices[y];
//	}
//private:
//	map<vertex_descriptor_t, ptrdiff_t> m_num_exposed_subvertices;
//};

vertex_descriptor_t find_blossom_tip(vertex_descriptor_t v, vertex_descriptor_t v, vector<vertex_descriptor_t> root, vector<vertex_descriptor_t> parent){
	vector<vertex_descriptor_t> augmenting_path_v = edmonds_find_augmenting_path(v, root[v], parent); // Finds path from v to root[v]
	vector<vertex_descriptor_t> augmenting_path_w = edmonds_find_augmenting_path(w, root[w], parent); // Finds path from w to root[w]
	
	//log_file << "\t(v,w) = (" << v << "," << w << ")." << endl;
	//print_augmenting_path("augmenting_path_v", augmenting_path_v, G_prime, mates, false);
	//print_augmenting_path("augmenting_path_w", augmenting_path_w, G_prime, mates, false);

	ptrdiff_t v_w_size_difference = augmenting_path_v.size() - augmenting_path_w.size();
	if( v_w_size_difference > 0 ){
		augmenting_path_v.erase( augmenting_path_v.begin(), augmenting_path_v.begin() + v_w_size_difference );
	}
	else if( v_w_size_difference < 0 ){
		augmenting_path_w.erase( augmenting_path_w.begin(), augmenting_path_w.begin() - v_w_size_difference );
	}

	if( v_w_size_difference != 0 ){
		//print_augmenting_path("augmenting_path_v (after making same size)", augmenting_path_v, G_prime, mates, false);
		//print_augmenting_path("augmenting_path_w (after making same size)", augmenting_path_w, G_prime, mates, false);
	}

	while( augmenting_path_v.front() != augmenting_path_w.front() && !augmenting_path_v.empty() ){
		augmenting_path_v.erase(augmenting_path_v.begin());
		augmenting_path_w.erase(augmenting_path_w.begin());
	}

	//print_augmenting_path("augmenting_path_v (after erasing up to start of blossom)", augmenting_path_v, G_prime, mates, false);
	//print_augmenting_path("augmenting_path_w (after erasing up to start of blossom)", augmenting_path_w, G_prime, mates, false);

	if( augmenting_path_v.empty() || augmenting_path_w.empty() ){
		cout_and_log_file << "ERROR! No common ancestor found for blossom." << endl;
		exit_program();
	}

	return augmenting_path_v.front();
}

vector<vertex_descriptor_t> BFS_matching(Graph& G_prime, map<vertex_descriptor_t, set<vertex_descriptor_t> > mates, map<vertex_descriptor_t, ptrdiff_t> num_exposed_subvertices){

	vector<vertex_descriptor_t> augmenting_path;

	///////////////////////////////////////////////////////////////////////////////////
	//
	// Initial data structures to keep track of information for implicit matching
	//
	vector<vertex_descriptor_t> parent( num_vertices(G_prime), graph_traits<Graph>::null_vertex() );
	vector<vertex_descriptor_t> root( num_vertices(G_prime), graph_traits<Graph>::null_vertex() );
	map<edge_descriptor_t, bool> marked_edges;
	//map<vertex_descriptor_t, bool> marked_vertices;
	//greater_than_by_num_exposed_subvertices compare(num_exposed_subvertices); // Allows sort by number of exposed subvertices, from high to low
	//set<vertex_descriptor_t, greater_than_by_num_exposed_subvertices> exposed_vertices( compare );	
	set<vertex_descriptor_t> exposed_vertices;	

	// Add exposed vertices to exposed_vertices
	vertex_iterator_t vi, vi_end;
	size_t total_num_exposed_subvertices = 0;	// Keep track of the total number of exposed subvertices, if only 1, exit
	for(boost::tie(vi,vi_end) = vertices(G_prime); vi != vi_end; ++vi){
		vertex_descriptor_t v = *vi;
		if( num_exposed_subvertices[v] >= 1 ){
			total_num_exposed_subvertices += num_exposed_subvertices[v];

			// Set the parent, root and level of the the vertex v (where (v,w) is the current edge) to null
			parent[v] = graph_traits<Graph>::null_vertex();
			root[v] = graph_traits<Graph>::null_vertex();

			// Set v as unmarked and add to F
			//marked_vertices[v] = false;
			exposed_vertices.insert(v);
		}
	}

	cout_and_log_file << "\tedmonds_implicit_matching:  Total number of exposed subvertices: " << total_num_exposed_subvertices << endl;
	if( total_num_exposed_subvertices <= 1 ){
		return augmenting_path;
	}

	bool augmenting_path_found = false;
	
	while( !exposed_vertices.empty() && !augmenting_path_found ){
		vertex_descriptor_t current_exposed_vertex = *exposed_vertices.begin();
		log_file << "Exposed vertex: " << current_exposed_vertex << endl;
		exposed_vertices.erase(exposed_vertices.begin());
		
		set<vertex_descriptor_t> current_vertices;
		// Find initial set of vertices adjacent to the exposed vertex
		out_edge_iterator_t ei, ei_end;
		for(tie(ei,ei_end) = out_edges(current_exposed_vertex, G_prime); ei != ei_end; ++ei){

			// Current edge info
			edge_descriptor_t current_edge = *ei;
			vertex_descriptor_t w = target(current_edge, G_prime);

			// Check if edge is marked, skip
			if( marked_edges[current_edge] ){
				continue;
			}
			marked_edges[current_edge] = true;

			// Check if edges is matched, skip
			if( mates[current_exposed_vertex].find(w) != mates[current_exposed_vertex].end() ){ // True if current edge is matched
				continue;
			}

			// Check if w is exposed, return edge between them as 
			if( exposed_vertices.find(w) != exposed_vertices.end() ){
				augmenting_path.push_back(current_exposed_vertex);
				augmenting_path.push_back(w);

				augmenting_path_found = true;
				break;
			}

			// Set the parent, root and level of the the vertex w (where (v,w) is the current edge) to w
			parent[w] = current_exposed_vertex;
			root[w] = w;	// Take the root to be the first set of adjacent edges to the current exposed vertex, not the current exposed vertex itself

			current_vertices.insert(w);
		}

		// BFS on the vertices, with each level checking for either matched or unmatched edges
		ptrdiff_t level = 1;
		bool currently_matched = true;	// If true, we are currently looking for matched edges, else unmatched edges
		while( !current_vertices.empty() && !augmenting_path_found ){
			log_file << "Level " << level++ << endl;
			print_degree_sequence("Vertices", current_vertices);
			log_file << endl;

			set<vertex_descriptor_t> next_vertices;
	
			while( !current_vertices.empty() && !augmenting_path_found ){
				vertex_descriptor_t v = *current_vertices.begin();
				current_vertices.erase(current_vertices.begin());

				out_edge_iterator_t ei, ei_end;
				for(tie(ei,ei_end) = out_edges(v, G_prime); ei != ei_end; ++ei){

					// Current edge info
					edge_descriptor_t current_edge = *ei;
					vertex_descriptor_t w = target(current_edge, G_prime);

					// Check if edge is marked, skip
					if( marked_edges[current_edge] ){
						continue;
					}
					marked_edges[current_edge] = true;

					// Check if w is exposed and current edge is matched, if so, ignore, as these pathways can be found for those vertices in future iterations
					if( currently_matched && exposed_vertices.find(w) != exposed_vertices.end() ){
						continue;
					}

					// Check that current edge is matched or unmatched, depending on currently_matched
					if( currently_matched && mates[v].find(w) == mates[v].end() ){ // True if current edge isn't matched, but should be
						continue;
					}
					if( !currently_matched && mates[v].find(w) != mates[v].end() ){ // True if current edge is matched, but shouldn't be
						continue;
					}

					// Check if w is exposed and (v,w) is unmatched, find path from w to current exposed vertex
					if( !currently_matched && exposed_vertices.find(w) != exposed_vertices.end() ){
						// Find augmenting path
						augmenting_path = edmonds_find_augmenting_path(v, root[v], parent);
						augmenting_path.push_back(current_exposed_vertex);
						reverse(augmenting_path.begin(), augmenting_path.end());
						augmenting_path.push_back(w);

						augmenting_path_found = true;
						break;
					}

					// Check if w forms a blossom that starts at the current exposed vertex (assuming the current exposed vertex has 2 or more exposed subvertices)
					if( num_exposed_subvertices[current_exposed_vertex] >= 2 ){
						// If the blossom starts at the current exposed vertex
						if( root[v] != root[w] && root[w] != graph_traits<Graph>::null_vertex() ){
							if( w != parent[w] ){
								// If (v,w) is the same as (w,parent[w]) in terms of being matched or unmatched, ignore
								if( (currently_matched && is_matched(w, parent[w], mates)) || (!currently_matched && !is_matched(w, parent[w], mates)) ){
									continue;
								}
							}

							log_file << "Blossom starting at exposed vertex." << endl;

							// Find augmenting path
							vector<vertex_descriptor_t> augmenting_path_v = edmonds_find_augmenting_path(v, root[v], parent);
							vector<vertex_descriptor_t> augmenting_path_w = edmonds_find_augmenting_path(w, root[w], parent);
							augmenting_path_v.push_back(current_exposed_vertex);
							augmenting_path_w.push_back(current_exposed_vertex);
							reverse(augmenting_path_v.begin(), augmenting_path_v.end());

							augmenting_path.assign(augmenting_path_v.begin(), augmenting_path_v.end());
							augmenting_path.insert(augmenting_path.end(), augmenting_path_w.begin(), augmenting_path_w.end());

							augmenting_path_found = true;
							break;
						}
					}

					// Set the parent, root and level of the the vertex v (where (v,w) is the current edge)
					if( parent[w] == graph_traits<Graph>::null_vertex() ){
						parent[w] = v;
						root[w] = root[v];
					}

					next_vertices.insert(w);
				}
			}
			
			if( !augmenting_path_found ){
				currently_matched = !currently_matched;
				current_vertices = next_vertices;
			}
		}
	}	

	print_augmenting_path("Augmenting path P", augmenting_path, G_prime, mates);
	return augmenting_path;
}

vector<vertex_descriptor_t> DFS_matching(Graph& G_prime, map<vertex_descriptor_t, set<vertex_descriptor_t> > mates, map<vertex_descriptor_t, ptrdiff_t> num_exposed_subvertices){

	///////////////////////////////////////////////////////////////////////////////////
	//
	// Initial data structures to keep track of information for implicit matching
	//
	vector<vertex_descriptor_t> augmenting_path;
	map<edge_descriptor_t, bool> marked_edges;
	set<vertex_descriptor_t> exposed_vertices;

	// Add exposed vertices to exposed_vertices
	vertex_iterator_t vi, vi_end;
	size_t total_num_exposed_subvertices = 0;	// Keep track of the total number of exposed subvertices, if only 1, exit
	for(boost::tie(vi,vi_end) = vertices(G_prime); vi != vi_end; ++vi){
		vertex_descriptor_t v = *vi;
		if( num_exposed_subvertices[v] >= 1 ){
			total_num_exposed_subvertices += num_exposed_subvertices[v];
			exposed_vertices.insert(v);
		}
	}

	cout_and_log_file << "\tedmonds_implicit_matching:  Total number of exposed subvertices: " << total_num_exposed_subvertices << endl;
	if( total_num_exposed_subvertices <= 1 ){
		return augmenting_path;
	}

	bool augmenting_path_found = false;
	while( !exposed_vertices.empty() && !augmenting_path_found ){
		vertex_descriptor_t current_exposed_vertex = *exposed_vertices.begin();
		log_file << "Exposed vertex: " << current_exposed_vertex << endl;
		
		augmenting_path_found = DFS_recursion(G_prime, current_exposed_vertex, false, augmenting_path, exposed_vertices, num_exposed_subvertices, marked_edges, mates);

		exposed_vertices.erase(exposed_vertices.begin());
	}	

	print_augmenting_path("Augmenting path P", augmenting_path, G_prime, mates);	
	return augmenting_path;
}

bool DFS_recursion(Graph& G_prime, vertex_descriptor_t current_vertex, bool currently_matched, vector<vertex_descriptor_t>& augmenting_path, set<vertex_descriptor_t> exposed_vertices, map<vertex_descriptor_t, ptrdiff_t> num_exposed_subvertices, map<edge_descriptor_t, bool>& marked_edges, map<vertex_descriptor_t, set<vertex_descriptor_t> > mates ){
	augmenting_path.push_back( current_vertex );

	// DFS on the vertices, with each level checking for either matched or unmatched edges	
	out_edge_iterator_t ei, ei_end;
	for(tie(ei,ei_end) = out_edges(current_vertex, G_prime); ei != ei_end; ++ei){	

		// Check if edge is marked, skip
		if( marked_edges[*ei] ){
			continue;
		}
		marked_edges[*ei] = true;

		vertex_descriptor_t w = target(*ei, G_prime);



		// Check that current edge is matched or unmatched, depending on currently_matched
		if( currently_matched && !is_matched(current_vertex, w, mates) ){ // True if current edge isn't matched, but should be
			continue;
		}
		if( !currently_matched && is_matched(current_vertex, w, mates) ){ // True if current edge is matched, but shouldn't be
			continue;
		}

		// Check if w is exposed and current edge is matched, if so, ignore, as these pathways can be found for those vertices in future iterations
		if( exposed_vertices.find(w) != exposed_vertices.end() ){
			if( currently_matched ){
				continue;
			}
			// Check if w is exposed and (v,w) is unmatched, return augmenting path
			else{
				if( augmenting_path.front() == w && num_exposed_subvertices[w] < 2 ){
					continue;
				}
				augmenting_path.push_back(w);
				return true;
			}
		}

		if( DFS_recursion(G_prime, w, !currently_matched, augmenting_path, exposed_vertices, num_exposed_subvertices, marked_edges, mates) ){
			return true;
		}
	}
	augmenting_path.pop_back();
	return false;
}

bool is_matched(vertex_descriptor_t u, vertex_descriptor_t v, map<vertex_descriptor_t, set<vertex_descriptor_t> > mates){
	return mates[u].find(v) != mates[u].end();
}

size_t greedy_implicit_initial_matching(Graph G_prime, map<vertex_descriptor_t, set<vertex_descriptor_t> > &mates, map<vertex_descriptor_t, ptrdiff_t> &num_exposed_subvertices){
	edge_iterator_t ei, ei_end;
	size_t initial_anonymized_cost_handled = 0;	// Keep track of the anonymized cost handled by initial matching
	for( tie(ei, ei_end) = edges(G_prime); ei != ei_end; ++ei){
		edge_descriptor_t e = *ei;
		vertex_descriptor_t u = source(e, G_prime);
		vertex_descriptor_t v = target(e, G_prime);

		if(num_exposed_subvertices[u] >= 1 && num_exposed_subvertices[v] >= 1){
			pair< set<vertex_descriptor_t>::iterator, bool > ret;

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
//  and has minimum degree over all of v's neighbors. Add (u,v) to the matching. Ties for either 
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

size_t extra_greedy_implicit_initial_matching(Graph G_prime, map<vertex_descriptor_t, set<vertex_descriptor_t> > &mates, map<vertex_descriptor_t, ptrdiff_t> &num_exposed_subvertices){
	vector< vertex_pair_t > edge_list;

	edge_iterator_t ei, ei_end;
	for( tie(ei, ei_end) = edges(G_prime); ei != ei_end; ++ei){
		edge_descriptor_t e = *ei;
		vertex_descriptor_t u = source(e, G_prime);
		vertex_descriptor_t v = target(e, G_prime);

		edge_list.push_back( make_pair(u,v) );
		edge_list.push_back( make_pair(v,u) );
	}

	// Sort the edges by the degree of the target, then (using a stable sort) by degree of the source
	sort(edge_list.begin(), edge_list.end(), less_than_by_degree<select_second>(G_prime) );
	stable_sort(edge_list.begin(), edge_list.end(), less_than_by_degree<select_first>(G_prime) );
      
	// Construct the extra greedy matching
	size_t initial_anonymized_cost_handled = 0;	// Keep track of the anonymized cost handled by initial matching
	for(vector< vertex_pair_t >::const_iterator it = edge_list.begin(); it != edge_list.end(); ++it)
	{
		if( num_exposed_subvertices[it->first] >= 1 && num_exposed_subvertices[it->second] >= 1 ){
			pair< set<vertex_descriptor_t>::iterator, bool > ret;

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
vector<vertex_descriptor_t> edmonds_find_augmenting_path(vertex_descriptor_t u, vertex_descriptor_t v, vector<vertex_descriptor_t> parent){
	vector<vertex_descriptor_t> augmenting_path_u;
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

void DEBUG_RECURSION_START(string debug_message){
	cout_and_log_file << endl << "\tStart (recursion level " << recursion_level++ << "): " << debug_message << endl << endl;
}

void DEBUG_RECURSION_END(string debug_message){
	cout_and_log_file << endl << "\tEnd (recursion level " << --recursion_level << "): " << debug_message << endl << endl;
}

void DEBUG_START(string debug_message){
	t.restart();
	cout_and_log_file << endl << "Start: " << debug_message << endl << endl;
}

void DEBUG_END(string debug_message){
	cout_and_log_file << endl << "End: " << debug_message << " (Took " << t.elapsed() << " seconds)" << endl << endl;
}

void DEBUG_START_L2(string debug_message){
	t_l2.restart();
	cout_and_log_file << endl << "\tStart: " << debug_message << endl << endl;
}

void DEBUG_END_L2(string debug_message){
	cout_and_log_file << endl << "\tEnd: " << debug_message << " (Took " << t_l2.elapsed() << " seconds)" << endl << endl;
}

void exit_program(){
	exit(1);
}

void select_input_graph(size_t& input_graph){
	///////////////////////////////////////////////////////////////////////////////////
	//
	// Get input graph choice
	//
	string input = "";
	while(1){
		cout_and_log_file << endl << "Enter input graph, " << SMALL_WORLD_GRAPH << " for " << graph_titles[SMALL_WORLD_GRAPH] 
			<< ", " << ENRON_GRAPH << " for " << graph_titles[ENRON_GRAPH]
			<< ", " << KARATE_GRAPH << " for " << graph_titles[KARATE_GRAPH]
			<< "(0 to EXIT, ENTER for default<" << graph_titles[default_input_graph] << ">): ";
		cout_and_log_file.flush();
		getline(cin, input);

		if( input.empty() ){
			input_graph = default_input_graph;
			break;
		}

		// This code converts from string to number safely.
		stringstream myStream(input);

		if (myStream >> input_graph){
			if( input_graph == 0 ){
				exit_program();
			}
			else if( input_graph != SMALL_WORLD_GRAPH && input_graph != ENRON_GRAPH && input_graph != KARATE_GRAPH ){
				cout_and_log_file << "Invalid number, must be either " << SMALL_WORLD_GRAPH << ", " << ENRON_GRAPH << ", or " << KARATE_GRAPH << endl;
				continue;
			}
			else{
				default_input_graph = input_graph;
				break;
			}
		}
		cout_and_log_file << "Invalid number, must be either " << SMALL_WORLD_GRAPH << ", " << ENRON_GRAPH << ", or " << KARATE_GRAPH << endl;
	}
	cout_and_log_file << "You chose: " << graph_titles[input_graph] << endl << endl;
}

void get_inputs(size_t& number_of_experiments, size_t& k, double& subset_X_percent){

	///////////////////////////////////////////////////////////////////////////////////
	//
	// Get number of experiments to run, check if valid
	//
	string input = "";
	while(1){
		cout_and_log_file << endl << "Enter number of experiments to run (0 to EXIT, ENTER for default<" << default_number_of_experiments << ">): ";
		cout_and_log_file.flush();
		getline(cin, input);

		if( input.empty() ){
			number_of_experiments = default_number_of_experiments;
			break;
		}

		// This code converts from string to number safely.
		stringstream myStream(input);

		if (myStream >> number_of_experiments){
			if( number_of_experiments <= 0 ){
				if( number_of_experiments == 0 ){
					exit_program();
				}
				else{
					cout_and_log_file << "Invalid number, must be greater than 0" << endl;
					continue;
				}
			}
			else{
				default_number_of_experiments = number_of_experiments;
				break;
			}
		}
		cout_and_log_file << "Invalid number, please try again" << endl;
	}
	cout_and_log_file << "You entered: " << number_of_experiments << endl << endl;

	///////////////////////////////////////////////////////////////////////////////////
	//
	// Get k value, check if valid
	//
	input = "";
	while(1){
		cout_and_log_file << "Enter value for k (0 to EXIT, ENTER for default<" << default_k << ">): ";
		cout_and_log_file.flush();
		getline(cin, input);

		if( input.empty() ){
			k = default_k;
			break;
		}

		// This code converts from string to number safely.
		stringstream myStream(input);

		if (myStream >> k){
			if( k <= 1 ){
				if( k == 0 ){
					exit_program();
				}
				else{
					cout_and_log_file << "Invalid number, must be greater than 1" << endl;
					continue;
				}
			}
			else{
				default_k = k;
				break;
			}
		}
		cout_and_log_file << "Invalid number, please try again" << endl;
	}
	cout_and_log_file << "You entered: " << k << endl << endl;

	///////////////////////////////////////////////////////////////////////////////////
	//
	// Get subset_X_percent value, check if valid
	//
	input = "";
	while(1){
		cout_and_log_file << "Enter value for subset X percent of G (0 to EXIT, ENTER for default<" << default_subset_X_percent << ">): ";
		cout_and_log_file.flush();
		getline(cin, input);

		if( input.empty() ){
			subset_X_percent = default_subset_X_percent;
			break;
		}

		// This code converts from string to number safely.
		stringstream myStream(input);

		if (myStream >> subset_X_percent){
			if( subset_X_percent <= 0.0){
				exit_program();
			}
			else if( subset_X_percent > 1.0 ){
				cout_and_log_file << "Invalid number, must be greater than 0 and less than or equal to 1.0" << endl;
				continue;
			}
			else{
				default_subset_X_percent = subset_X_percent;
				break;
			}
		}
		cout_and_log_file << "Invalid number, please try again" << endl;
	}
	cout_and_log_file << "You entered: " << subset_X_percent << endl << endl;
}

void get_inputs(size_t& number_of_experiments, size_t& k, double& subset_X_percent, size_t& number_of_vertices, size_t& k_nearest_neighbors){

	// get k and subset_X_percent from user
	get_inputs(number_of_experiments, k, subset_X_percent);

	///////////////////////////////////////////////////////////////////////////////////
	//
	// Get number of vertices, check if valid
	//
	string input = "";
	while(1){
		cout_and_log_file << endl << "Enter number of vertices (0 to EXIT, ENTER for default<" << default_number_of_vertices << ">): ";
		cout_and_log_file.flush();
		getline(cin, input);

		if( input.empty() ){
			number_of_vertices = default_number_of_vertices;
			break;
		}

		// This code converts from string to number safely.
		stringstream myStream(input);

		if (myStream >> number_of_vertices){
			if( number_of_vertices <= 1 ){
				if( number_of_vertices == 0 ){
					exit_program();
				}
				else{
					cout_and_log_file << "Invalid number, must be greater than 1" << endl;
					continue;
				}
			}
			else{
				default_number_of_vertices = number_of_vertices;
				break;
			}
		}
		cout_and_log_file << "Invalid number, please try again" << endl;
	}
	cout_and_log_file << "You entered: " << number_of_vertices << endl << endl;

	///////////////////////////////////////////////////////////////////////////////////
	//
	// Get k_nearest_neighbors_percent value, check if valid
	//
	input = "";
	while(1){
		cout_and_log_file << "Each vertex connected to its k-nearest neighbors in small-world graph.  Enter value for k-nearest neighbors (0 to EXIT, ENTER for default<" << default_k_nearest_neighbors << ">): ";
		cout_and_log_file.flush();
		getline(cin, input);

		if( input.empty() ){
			k_nearest_neighbors = default_k_nearest_neighbors;
			break;
		}

		// This code converts from string to number safely.
		stringstream myStream(input);

		if (myStream >> k_nearest_neighbors){
			if( k_nearest_neighbors < 1 || k_nearest_neighbors >= number_of_vertices ){
				if( k_nearest_neighbors == 0 ){
					exit_program();
				}
				else{
					cout_and_log_file << "Invalid number, must be between 1 and " << number_of_vertices - 1 << endl;
					continue;
				}
			}
			else{
				default_k_nearest_neighbors = k_nearest_neighbors;
				break;
			}
		}
		cout_and_log_file << "Invalid number, please try again" << endl;
	}
	cout_and_log_file << "You entered: " << k_nearest_neighbors << endl << endl;
	cout_and_log_file.flush();
}

// Print augmenting path (or blossom)
void print_augmenting_path(const string& description, vector<vertex_descriptor_t> augmenting_path, Graph G_prime, map<vertex_descriptor_t, set<vertex_descriptor_t> > mates, bool output_to_cout_and_error_check ){	
	// Print out vector
	vector<vertex_descriptor_t>::iterator it;	
	if( output_to_cout_and_error_check ){
		cout_and_log_file << endl;
		cout_and_log_file << description << ": " << endl;
		bool matched = false; // Use to check that it is an alternating path
		bool error = false;
		if( augmenting_path.size() >= 2 ){
			for(it = augmenting_path.begin(); it < augmenting_path.end()-1; ++it){
				if( mates[it[0]].find(it[1]) != mates[it[0]].end() ){ // True if there's a matching
					if( !matched ) error = true;
					cout_and_log_file << *it << " xxx "; // Matching
					matched = false; // Check that next edge is not matched
					
				}
				else{
					if( matched ) error = true;
					if( edge(it[0], it[1], G_prime).second ) {	// True an edge exists between the two
						cout_and_log_file << *it << " --- ";
					}
					else{
						cout_and_log_file << *it << " -no edge- ";
						error = true;
					}
					matched = true;	// Check that the next edge is matched
				}
			}
			cout_and_log_file << *it << endl << endl;
		}
		else if( augmenting_path.size() == 1){
			cout_and_log_file << augmenting_path.front() << endl << endl;
		}
		else{
			cout_and_log_file << "Augmenting path is empty." << endl << endl;
		}

		if( error ){
			cout_and_log_file << "ERROR!!! Path is not augmenting." << endl << endl;
		}
	}
	else{
		log_file << endl;
		log_file << description << ": " << endl;
		if( augmenting_path.size() >= 2 ){
			for(it = augmenting_path.begin(); it < augmenting_path.end()-1; ++it){
				if( mates[it[0]].find(it[1]) != mates[it[0]].end() ){
					log_file << *it << " xxx "; // Matching
				}
				else{
					log_file << *it << " --- ";
				}
			}
			log_file << *it << endl << endl;
		}
		else if( augmenting_path.size() == 1){
			log_file << augmenting_path.front() << endl << endl;
		}
		else{
			log_file << "Augmenting path is empty." << endl << endl;
		}
	}
}

void print_degree_sequence(const string& description, vector<size_t> d){
	log_file << endl;
	log_file << description << ": " << endl;
	vector<size_t>::iterator it;
	for(it = d.begin(); it < d.end()-1; ++it){
		log_file << *it << "\t";
	}
	log_file << *it << endl;
}

void print_degree_sequence(const string& description, vector<degree_vertex_pair> d){
	log_file << endl;
	log_file << description << " (vertex id below in brackets): " << endl;
	vector<degree_vertex_pair>::iterator it;
	for(it = d.begin(); it < d.end()-1; ++it){
		log_file << (*it).first << "\t";
	}
	log_file << (*it).first << endl;
	for(it = d.begin(); it < d.end()-1; ++it){
		log_file << "(" << (*it).second << ")" << "\t";
	}
	log_file << "(" << (*it).second << ")" << endl;
}

void print_degree_sequence(const string& description, map<vertex_descriptor_t, ptrdiff_t> d, bool output_to_cout){
	if( output_to_cout ){
		cout_and_log_file << endl;
		cout_and_log_file << description << " (vertex id below in brackets): " << endl;
		map<vertex_descriptor_t, ptrdiff_t>::iterator mit;
		vector<vertex_descriptor_t> vertex_numbers;
		for(mit = d.begin(); mit != d.end(); ++mit){
			mit++;
			if( mit == d.end() ){
				mit--;
				if( (*mit).second != 0 ){
					cout_and_log_file << (*mit).second << endl;
					vertex_numbers.push_back( (*mit).first );
				}
			}
			else{
				mit--;
				if( (*mit).second != 0 ){
					cout_and_log_file << (*mit).second << "\t";
					vertex_numbers.push_back( (*mit).first );
				}
			}
			
		}
		cout_and_log_file << endl;
		if( !vertex_numbers.empty() ){
			vector<size_t>::iterator vit;
			for(vit = vertex_numbers.begin(); vit < vertex_numbers.end()-1; ++vit){
				cout_and_log_file << "(" << *vit << ")" << "\t";
			}
			cout_and_log_file << "(" << *vit << ")" << endl;
		}
		else{
			cout_and_log_file << "All values are 0." << endl;
		}
	}
	else{
		log_file << endl;
		log_file << description << " (vertex id below in brackets): " << endl;
		map<vertex_descriptor_t, ptrdiff_t>::iterator mit;
		vector<vertex_descriptor_t> vertex_numbers;
		for(mit = d.begin(); mit != d.end(); ++mit){
			mit++;
			if( mit == d.end() ){
				mit--;
				if( (*mit).second != 0 ){
					log_file << (*mit).second << endl;
					vertex_numbers.push_back( (*mit).first );
				}
			}
			else{
				mit--;
				if( (*mit).second != 0 ){
					log_file << (*mit).second << "\t";
					vertex_numbers.push_back( (*mit).first );
				}
			}
			
		}
		log_file << endl;
		if( !vertex_numbers.empty() ){
			vector<size_t>::iterator vit;
			for(vit = vertex_numbers.begin(); vit < vertex_numbers.end()-1; ++vit){
				log_file << "(" << *vit << ")" << "\t";
			}
			log_file << "(" << *vit << ")" << endl;
		}
		else{
			log_file << "All values are 0." << endl;
		}
	}
}

void print_degree_sequence(const string& description, set<vertex_descriptor_t> d){
	log_file << endl;
	log_file << description << ": " << endl;
	set<vertex_descriptor_t>::iterator it;
	for(it = d.begin(); it != d.end(); ++it){
		log_file << *it << "\t";
	}
	log_file << endl;
}

bool is_k_degree_anonymous(vector<degree_vertex_pair> degree_sequence, size_t k){
	vector<degree_vertex_pair>::iterator it = degree_sequence.begin();
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
size_t ErdosGallaiThm(vector< vector<degree_vertex_pair> > SetOfDegreeSequences, size_t n)
{
	vector<degree_vertex_pair>::iterator it;
	vector< vector<degree_vertex_pair> >::iterator it2;
	for(it2 = SetOfDegreeSequences.begin(); it2 < SetOfDegreeSequences.end(); it2++ ){
		for(size_t j = 1; j <= n; j++){
			size_t sum = 0;
			for(size_t i = 1; i <= j; i++){
				sum += (*it2).at(i-1).first;
			}

			size_t sum_2 = j*(j-1);
			for(size_t i = j+1; i <= n; i++){
				sum_2 += min((*it2).at(i-1).first,j);
			}

			if( sum > sum_2 ){
				cout_and_log_file << "Degree sequence does not represent a real graph, fails at d_" << j << ": ";
				for(it = (*it2).begin(); it < (*it2).end()-1; ++it){
					cout_and_log_file << (*it).first << ",";
				}
				cout_and_log_file << (*it).first << endl;
				cout_and_log_file << "Press ENTER to continue... " << flush;
				cin.ignore( numeric_limits<streamsize>::max(), '\n' );
				cout_and_log_file << endl << endl;
				return 0;
			}

			if( j == n && sum % 2 != 0 ){
				cout_and_log_file << "Degree sequence does not represent a real graph, sum of degrees is not even." << endl;
				cout_and_log_file << "Press ENTER to continue... " << flush;
				cin.ignore( numeric_limits<streamsize>::max(), '\n' );
				cout_and_log_file << endl << endl;
				return 0;
			}
		}
	}
	cout_and_log_file << endl << endl;
	return 1;
}

///////////////////////////////////////////////////////////////////////////////////
//
//	Determine the total number of possible k-groupings
//		Isn't quite correct, as should check for when degree values are the same and can be size_terchanged
//
size_t NumberOfKGroupings(size_t total, vector<degree_vertex_pair> d_reverse)
{
	vector<degree_vertex_pair> d_prime(d_reverse);
	switch(d_prime.size()){
		case 1:
			return total;
		case 2:
		case 3:
			return total + 1;
		default:
			vector<degree_vertex_pair>::iterator it;
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
size_t DAGroupCost(vector<degree_vertex_pair> d, size_t start, size_t end)
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
vector< vector<degree_vertex_pair> > DegreeAnonymization(vector<degree_vertex_pair> d, size_t number_of_vertices, size_t k, bool AllPossible = false)
{
	if(!AllPossible){
		vector<size_t> DA(number_of_vertices,0);
		vector<size_t> I(number_of_vertices,0);
		vector<degree_vertex_pair> DegreeSequence(d);
		vector<size_t> Split(number_of_vertices,0);
		vector< vector<degree_vertex_pair> > SetOfDegreeSequences;
		
		for(size_t i = 1; i <= number_of_vertices; i++){
			if( i < 2*k ){
				if( i > 1 ){
					I[i-1] += I[i-2] + d.at(0).first - d.at(i-1).first;
					DA[i-1] = I.at(i-1);
				}
			}
			else{
				size_t min = numeric_limits<size_t>::max();
				size_t t_opt = 0;
				for(size_t t = max(k,i-2*k+1); t <= i-k; t++){
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
		vector<degree_vertex_pair>::iterator it;
		cout_and_log_file << "Cost of anonymizing: " << DA[number_of_vertices - 1] << endl;

		SetOfDegreeSequences.push_back(DegreeSequence);
		return SetOfDegreeSequences;
	}
	else{
		vector<size_t> DA(number_of_vertices,0);
		vector<size_t> I(number_of_vertices,0);
		vector<degree_vertex_pair> DegreeSequence(number_of_vertices);
		vector< vector<degree_vertex_pair> > SetOfPossibleDegreeSequences;
		vector< vector< vector<degree_vertex_pair> > > SetOfDegreeSequences;

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
				size_t min = numeric_limits<size_t>::max();
				for(size_t t = max(k,i-2*k+1); t <= i-k; t++){
					size_t cost = DA[t-1] + DAGroupCost(d,t+1,i);
					if( cost <= min ){
						if(cost != min){
							SetOfPossibleDegreeSequences.clear();
							min = cost;
						}
						vector< vector<degree_vertex_pair> >::iterator it;
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

		cout_and_log_file << "Cost of anonymizing: " << DA[number_of_vertices - 1] << endl;
		//vector<degree_vertex_pair>::iterator it;
		//vector< vector<degree_vertex_pair> >::iterator it2;
		//for(it2 = SetOfDegreeSequences.back().begin(); it2 < SetOfDegreeSequences.back().end(); it2++ ){
		//	cout_and_log_file << "Anonymized degree sequence: ";
		//	for(it = (*it2).begin(); it < (*it2).end()-1; ++it){
		//		cout_and_log_file << (*it).first << ",";
		//	}
		//	cout_and_log_file << (*it).first << endl;
		//}
		return SetOfDegreeSequences.back();
	}
}