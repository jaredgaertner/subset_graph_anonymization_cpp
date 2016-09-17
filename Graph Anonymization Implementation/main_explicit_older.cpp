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
#include <boost/graph/graphml.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/small_world_generator.hpp>
#include <boost/graph/max_cardinality_matching.hpp>
#include <boost/random.hpp>
#include <boost/timer.hpp>

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

typedef graph_traits<Graph>::vertex_descriptor vertex_desc;
typedef graph_traits<Graph>::vertex_iterator vertex_iterator;
typedef graph_traits<Graph>::edge_descriptor edge_desc;
typedef graph_traits<Graph>::out_edge_iterator edge_iterator;
typedef graph_traits<Graph>::adjacency_iterator adjacency_it;

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
typedef pair<size_t,vertex_desc> degree_vertex_pair;

// Compare for sort creating a descending order
bool compare_descending(degree_vertex_pair i, degree_vertex_pair j){ return i.first > j.first; }

// Output to cout and file log.txt at same time using boost tee
typedef boost::iostreams::tee_device< std::ostream, iostreams::stream<iostreams::file_sink> > tee_device;
typedef boost::iostreams::stream<tee_device> tee_stream;
iostreams::stream<iostreams::file_sink> log_file;
tee_device tee(cout, log_file);
tee_stream cout_and_log(tee);

// Functions
void exit_program();
void select_input_graph(size_t& input_graph);
void get_inputs(size_t& k, double& subset_X_percent);
void get_inputs(size_t& number_of_experiments, size_t& k, double& subset_X_percent, size_t& number_of_vertices, size_t& k_nearest_neighbors);
void print_t_degree_sequence(const string& description, vector<size_t> d);
void print_degree_sequence(const string& description, vector<degree_vertex_pair> d);
void print_degree_sequence(const string& description, map<vertex_desc,size_t> d);
bool is_k_degree_anonymous(vector<degree_vertex_pair> degree_sequence, size_t k);
size_t ErdosGallaiThm(vector< vector<size_t> > SetOfDegreeSequences, size_t n);
size_t NumberOfKGroupings(size_t total, vector<size_t> d_reverse);
size_t DAGroupCost(vector<size_t> d, size_t start, size_t end);
vector< vector<degree_vertex_pair> > DegreeAnonymization(vector<degree_vertex_pair> d, size_t number_of_vertices, size_t k, bool AllPossible);
vector< pair<vertex_desc, vertex_desc> > upper_degree_constrained_subgraph(Graph G, vector<degree_vertex_pair> d, map<vertex_desc,size_t> upper_bounds, map<vertex_desc,size_t> delta, size_t number_of_vertices_G_prime, size_t number_of_edges_G_prime, size_t number_of_vertices_G_prime_actual, size_t number_of_edges_G_prime_actual);

// Debug functions
timer t;
void DEBUG_START(string);
void DEBUG_END(string);

// Initial maximum matching
template <typename Graph, typename MateMap>
struct input_initial_matching
{ 
	typedef typename graph_traits<Graph>::vertex_iterator vertex_iterator_t;

	static void find_matching(const Graph& g, MateMap mate){}
};

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
		ss << "log_" << posix_time::second_clock::local_time() << ".txt";
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
				get_inputs(k, subset_X_percent);
				number_of_vertices = 36692;
				break;

			case KARATE_GRAPH:
				get_inputs(k, subset_X_percent);
				number_of_vertices = 34;
				break;

			default:
				cout_and_log << "Error: no graph option selected?" << endl;
		}

		// Keep track of the current experiment number and how many times the experiments have succeeded and failed
		size_t current_experiment = 1;
		size_t successes = 0, failures = 0;

		// Start looping until number_of_experiments have been run
		while( current_experiment <= number_of_experiments ){
			cout_and_log << "Experiment number: " << current_experiment << endl;

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
					cout_and_log << "Number of vertices: " << number_of_vertices << endl;
					cout_and_log << "Each connected to its " << k_nearest_neighbors << " nearest neighbors." << endl;
					cout_and_log << "Edges in the graph are randomly rewired to different vertices with a probability " << probability << " and self-loops are set to " << allow_self_loops << "." << endl;
					SmallWorldGenerator smg_it;
					for(smg_it = SmallWorldGenerator(gen, number_of_vertices, k_nearest_neighbors, probability, allow_self_loops); smg_it != SmallWorldGenerator(); smg_it++){
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
										cout_and_log << "Read " << number_of_edges_read << " edges out of 183831." << endl;
									}
								}
							}
						}
						input_file.close();
					}
					else{
						cout_and_log << "Unable to open file" << endl;
					}
					DEBUG_END("Reading in enron data ...");

					DEBUG_START("Creating graph from enron data ...");
					vector< pair<size_t,size_t> >::iterator eit;
					for(eit = edge_list.begin(); eit < edge_list.end(); eit++){
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
						cout_and_log << "Unable to open file" << endl;
					}
					DEBUG_END("Reading in karate data ...");

					DEBUG_START("Creating graph from karate data ...");
					vector< pair<size_t,size_t> >::iterator eit;
					for(eit = edge_list.begin(); eit < edge_list.end(); eit++){
						add_edge( (*eit).first, (*eit).second, G );
					}
					DEBUG_END("Creating graph from karate data ...");
					}
					break;

				default:
					cout_and_log << "Error: no graph option selected?" << endl;
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

			///////////////////////////////////////////////////////////////////////////////////
			//
			// Find a small subset of vertices, X, of G
			//
			DEBUG_START("Creating subset X of G ...");
			set<vertex_desc> X_vertices;
			if( subset_X_percent <= 0.99 ){
				while(X_vertices.size() < subset_X_percent * num_vertices(G)){
					// randomly add vertices to X
					double rand1 = uni01();
					size_t add_vertex_num = (size_t)(rand1 * num_vertices(G));
					X_vertices.insert( vertex(add_vertex_num, G) );
				}

				set<vertex_desc>::iterator sit;
				cout_and_log << "Vertices chosen for X from G: " << endl;
				for(sit = X_vertices.begin(); sit != X_vertices.end(); sit++){
					cout_and_log << *sit << ",";
				}
				cout_and_log << endl << endl;
			}
			else{
				vertex_iterator ui, ui_end, vi;
				for(tie(ui, ui_end) = vertices(G); ui != ui_end; ++ui){
					X_vertices.insert(*ui);
				}
			}

			SubGraph& X_sub = G.create_subgraph(X_vertices.begin(), X_vertices.end());
			Graph X;
			copy_graph(X_sub, X);
			size_t number_of_vertices_X = num_vertices(X);
			cout_and_log << "Number of vertices in X: " << number_of_vertices_X << endl;
			DEBUG_END("Creating subset X of G ...");

			// Find degree sequence
			vector< degree_vertex_pair > d;
			vertex_iterator ui, ui_end, vi;
			for(tie(ui, ui_end) = vertices(X); ui != ui_end; ++ui){
				//degree_vertex_pair degree_vertex( degree( *ui,X ), *ui ); // Anonymize induced X
				degree_vertex_pair degree_vertex( degree(X_sub.local_to_global(*ui), G), *ui ); // Anonymize X wrt G
				d.push_back(degree_vertex);
			}
			sort(d.begin(), d.end(), compare_descending);
			print_degree_sequence("Degree sequence", d);
			
			// Check for clique
			if( d.front().first == d.back().first ){
				cout_and_log << "Clique or no edges, skipping." << endl;
				continue;
			}

			/*
			// Determine total number of k-groupings
			size_t total = 0;
			vector<size_t> d_reverse(d.rbegin(), d.rend());
			total = NumberOfKGroupings(total, d_reverse);
			cout_and_log << "The total number of k-groupings: " << total << endl;
			*/
			

			///////////////////////////////////////////////////////////////////////////////////
			//
			// Degree Anonymization (Lui and Terzi 2008)
			//
			DEBUG_START("Determining degree anonymization ...");
			vector< vector<degree_vertex_pair> > SetOfDegreeSequences;
			SetOfDegreeSequences = DegreeAnonymization(d, number_of_vertices_X, k, false);
			vector<degree_vertex_pair> AnonymizedDegreeSequence(SetOfDegreeSequences.back());
			DEBUG_END("Determining degree anonymization ...");

			// Check if it is a real degree sequence
			// ErdosGallaiThm(SetOfDegreeSequences.back(), number_of_vertices);

			///////////////////////////////////////////////////////////////////////////////////
			//
			// Find lower and upper bounds and delta
			//
			DEBUG_START("Determine the upper bounds for vertices in X");
			map<vertex_desc,size_t> upper_bounds;
			vector<degree_vertex_pair>::iterator dvit;
			size_t anonymized_degree_sequence_index = 0;
			size_t number_of_vertices_X_prime = 0;
			size_t number_of_edges_X_prime = 0; // num_edges(X_com); // if including only internal to external edges, use 0, for all edges use num_edges(X_com)
			for(dvit = d.begin(); dvit != d.end(); dvit++){
				vertex_desc u = (*dvit).second;
				size_t d_i = number_of_vertices_X - 1 - degree(u, X);
				size_t u_i = AnonymizedDegreeSequence.at(anonymized_degree_sequence_index++).first - (*dvit).first;
				upper_bounds[u] = u_i;
				size_t delta_i = d_i - u_i;

				// Determine the number of edges that would be used if vertices with u_i == 0 weren't excluded
				number_of_vertices_X_prime += d_i + delta_i;
				number_of_edges_X_prime += d_i * delta_i;
			}
			print_degree_sequence("Upper bounds", upper_bounds);
			DEBUG_END("Determine the upper bounds for vertices in X");


			// Find the complement of the subset X
			DEBUG_START("Finding complement of X ...");
			//size_t number_of_edges_read_X = 0;
			//vector< pair<vertex_desc,vertex_desc> > edge_list_X_com( number_of_vertices_X * (number_of_vertices_X - 1) / 2 - num_edges(X) );
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
			map<vertex_desc,size_t> lower_bounds, delta;
			size_t number_of_vertices_X_prime_actual = 0, number_of_edges_X_prime_actual = 0;
			for(size_t i = 0; i < d.size(); i++){
				vertex_desc u = d.at(i).second;
				size_t d_i = degree(u, X_com);
				size_t u_i = upper_bounds[u];
				int delta_i = (int)(d_i - u_i);
				if( delta_i < 0 ){
					delta_i = 0;
				}
				delta[u] = delta_i;

				vertex_desc global_index_u = X_sub.local_to_global(u);
				size_t degree_in_X = degree(u, X_sub);
				size_t degree_in_G = degree(global_index_u, G);
				size_t degree_in_only_G = degree_in_G - degree_in_X;
				int l_i = (int)(upper_bounds[u] - degree_in_only_G);
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
			vector< pair<vertex_desc, vertex_desc> > H_star = upper_degree_constrained_subgraph(X_com, d, upper_bounds, delta, number_of_vertices_X_prime, number_of_edges_X_prime, number_of_vertices_X_prime_actual, number_of_edges_X_prime_actual);

			DEBUG_START("Adding edges from H to X, displaying final degree sequence ...");
			vector< pair<vertex_desc,vertex_desc> >::iterator vvit;
			for(vvit = H_star.begin(); vvit < H_star.end(); vvit++){
				add_edge( vertex((*vvit).first, X_sub), vertex((*vvit).second, X_sub), X_sub); // Add edge to X
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
			DEBUG_END("Adding edges from H to X, displaying final degree sequence ...");

			DEBUG_START("Find possible additional edges in G \\ X to make X k-anonymous...");
			// Find which vertices need more edges to become anonymized
			vector<degree_vertex_pair> unanonymized_vertices;
			size_t i = 0;
			for(dvit = d_added_edges_within_X.begin(); dvit < d_added_edges_within_X.end(); dvit++){
				size_t u_i = AnonymizedDegreeSequence.at(i++).first - (*dvit).first;
				if( u_i > 0 ){
					unanonymized_vertices.push_back( make_pair(u_i, (*dvit).second) );
				}
			}

			// If there are any unanonymized vertices, find possible edges to make X k-anonymous, otherwise exit noting success
			bool edges_found = true;
			if( unanonymized_vertices.size() > 0 ){
				sort(unanonymized_vertices.begin(), unanonymized_vertices.end(), compare_descending);

				// Find list of all possible nodes in G which can be connected to an unanonymized vertex in X
				for(dvit = unanonymized_vertices.begin(); dvit < unanonymized_vertices.end(); dvit++){
					size_t upper_bound = (*dvit).first;

					vertex_desc v = (*dvit).second;
					vertex_desc v_global = X_sub.local_to_global(v);
					vector<vertex_desc> possible_vertices;
					for(tie(ui, ui_end) = vertices(G); ui != ui_end; ++ui){
						if( !edge(*ui, v_global, G).second && !X_sub.find_vertex(*ui).second ){	// Check that there's no edge from u to v and u is not in X_final
							possible_vertices.push_back(*ui);
						}
					}

					size_t num_of_added_edges = 0;
					while(num_of_added_edges < upper_bound){
						// randomly add edges from G to X for unanonymized vertices in X
						double rand1 = uni01();
						if( possible_vertices.size() > 0 ){
							size_t random_index = (size_t)(rand1 * possible_vertices.size());
							add_edge(vertex(possible_vertices.at(random_index), G), v_global, G);
							cout_and_log << "Add edge {" << possible_vertices.at(random_index) << ", " << v_global << "} to X to make it k-degree anonymous." << endl;
							possible_vertices.erase(possible_vertices.begin() + random_index);
							num_of_added_edges++;
						}
						else{
							edges_found = false;
							break;
						}
					}

				}

				// Find degree sequence of X
				vector<degree_vertex_pair> d_final;
				for(tie(ui, ui_end) = vertices(X_sub); ui != ui_end; ++ui){
					degree_vertex_pair degree_vertex( degree(X_sub.local_to_global(*ui), G), *ui ); // Anonymize X wrt G
					d_final.push_back(degree_vertex);
				}
				sort(d_final.begin(), d_final.end(), compare_descending);
				print_degree_sequence("Degree sequence of final graph X", d_final);
			}
			else{
				cout_and_log << "No edges needed from G \\ X to X to make X k-anonymous." << endl;
			}
			DEBUG_END("Find possible edges to add from G \\ X to X to make X k-anonymous...");

			DEBUG_START("Check if X is k-anonymous...");
			if( edges_found ){
				cout_and_log << "SUCCESS: X made k-anonymous." << endl << endl;
				successes++;
			}
			else{
				cout_and_log << "FAIL: X not made k-anonymous." << endl << endl;
				write_graphviz(cout_and_log, G);
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
				cout_and_log << "G is k-anonymous." << endl << endl;
			}
			else{
				cout_and_log << "G is not k-anonymous." << endl << endl;
			}
			DEBUG_END("Check if G is k-anonymous...");

			current_experiment++;
			cout_and_log << "Succeeded " << successes << " times and failed " << failures << " times." << endl;
		}
	}
}

void DEBUG_START(string debug_message){
	t.restart();
	cout_and_log << endl << "Start: " << debug_message << endl << endl;
}

void DEBUG_END(string debug_message){
	cout_and_log << endl << "End: " << debug_message << " (Took " << t.elapsed() << " seconds)" << endl << endl;
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
		cout_and_log << endl << "Enter input graph, " << SMALL_WORLD_GRAPH << " for " << graph_titles[SMALL_WORLD_GRAPH] 
			<< ", " << ENRON_GRAPH << " for " << graph_titles[ENRON_GRAPH]
			<< ", " << KARATE_GRAPH << " for " << graph_titles[KARATE_GRAPH]
			<< "(0 to EXIT, ENTER for default<" << graph_titles[default_input_graph] << ">): ";
		cout_and_log.flush();
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
				cout_and_log << "Invalid number, must be either " << SMALL_WORLD_GRAPH << ", " << ENRON_GRAPH << ", or " << KARATE_GRAPH << endl;
				continue;
			}
			else{
				default_input_graph = input_graph;
				break;
			}
		}
		cout_and_log << "Invalid number, must be either " << SMALL_WORLD_GRAPH << ", " << ENRON_GRAPH << ", or " << KARATE_GRAPH << endl;
	}
	cout_and_log << "You chose: " << graph_titles[input_graph] << endl << endl;
}

void get_inputs(size_t& k, double& subset_X_percent){

	///////////////////////////////////////////////////////////////////////////////////
	//
	// Get k value, check if valid
	//
	string input = "";
	while(1){
		cout_and_log << "Enter value for k (0 to EXIT, ENTER for default<" << default_k << ">): ";
		cout_and_log.flush();
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
					cout_and_log << "Invalid number, must be greater than 1" << endl;
					continue;
				}
			}
			else{
				default_k = k;
				break;
			}
		}
		cout_and_log << "Invalid number, please try again" << endl;
	}
	cout_and_log << "You entered: " << k << endl << endl;

	///////////////////////////////////////////////////////////////////////////////////
	//
	// Get subset_X_percent value, check if valid
	//
	input = "";
	while(1){
		cout_and_log << "Enter value for subset X percent of G (0 to EXIT, ENTER for default<" << default_subset_X_percent << ">): ";
		cout_and_log.flush();
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
				cout_and_log << "Invalid number, must be greater than 0 and less than or equal to 1.0" << endl;
				continue;
			}
			else{
				default_subset_X_percent = subset_X_percent;
				break;
			}
		}
		cout_and_log << "Invalid number, please try again" << endl;
	}
	cout_and_log << "You entered: " << subset_X_percent << endl << endl;
}

void get_inputs(size_t& number_of_experiments, size_t& k, double& subset_X_percent, size_t& number_of_vertices, size_t& k_nearest_neighbors){
	///////////////////////////////////////////////////////////////////////////////////
	//
	// Get number of experiments to run, check if valid
	//
	string input = "";
	while(1){
		cout_and_log << endl << "Enter number of experiments to run (0 to EXIT, ENTER for default<" << default_number_of_experiments << ">): ";
		cout_and_log.flush();
		getline(cin, input);

		if( input.empty() ){
			number_of_experiments = default_number_of_experiments;
			break;
		}

		// This code converts from string to number safely.
		stringstream myStream(input);

		if (myStream >> number_of_experiments){
			if( number_of_experiments <= 1 ){
				if( number_of_experiments == 0 ){
					exit_program();
				}
				else{
					cout_and_log << "Invalid number, must be greater than 1" << endl;
					continue;
				}
			}
			else{
				default_number_of_experiments = number_of_experiments;
				break;
			}
		}
		cout_and_log << "Invalid number, please try again" << endl;
	}
	cout_and_log << "You entered: " << number_of_experiments << endl << endl;

	// get k and subset_X_percent from user
	get_inputs(k, subset_X_percent);

	///////////////////////////////////////////////////////////////////////////////////
	//
	// Get number of vertices, check if valid
	//
	input = "";
	while(1){
		cout_and_log << endl << "Enter number of vertices (0 to EXIT, ENTER for default<" << default_number_of_vertices << ">): ";
		cout_and_log.flush();
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
					cout_and_log << "Invalid number, must be greater than 1" << endl;
					continue;
				}
			}
			else{
				default_number_of_vertices = number_of_vertices;
				break;
			}
		}
		cout_and_log << "Invalid number, please try again" << endl;
	}
	cout_and_log << "You entered: " << number_of_vertices << endl << endl;

	///////////////////////////////////////////////////////////////////////////////////
	//
	// Get k_nearest_neighbors_percent value, check if valid
	//
	input = "";
	while(1){
		cout_and_log << "Each vertex connected to its k-nearest neighbors in small-world graph.  Enter value for k-nearest neighbors (0 to EXIT, ENTER for default<" << default_k_nearest_neighbors << ">): ";
		cout_and_log.flush();
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
					cout_and_log << "Invalid number, must be between 1 and " << number_of_vertices - 1 << endl;
					continue;
				}
			}
			else{
				default_k_nearest_neighbors = k_nearest_neighbors;
				break;
			}
		}
		cout_and_log << "Invalid number, please try again" << endl;
	}
	cout_and_log << "You entered: " << k_nearest_neighbors << endl << endl;
	cout_and_log.flush();
}

void print_degree_sequence(const string& description, vector<size_t> d){
	cout_and_log << description << ": " << endl;
	vector<size_t>::iterator it;
	for(it = d.begin(); it < d.end()-1; it++){
		cout_and_log << *it << ",";
	}
	cout_and_log << *it << endl;
}

void print_degree_sequence(const string& description, vector<degree_vertex_pair> d){
	cout_and_log << description << " (vertex id in brackets): " << endl;
	vector<degree_vertex_pair>::iterator it;
	for(it = d.begin(); it < d.end()-1; it++){
		cout_and_log << (*it).first << ",";
	}
	cout_and_log << (*it).first << endl;
	cout_and_log << "(";
	for(it = d.begin(); it < d.end()-1; it++){
		cout_and_log << (*it).second << ",";
	}
	cout_and_log << (*it).second << ")" << endl;
}

void print_degree_sequence(const string& description, map<vertex_desc,size_t> d){
	cout_and_log << description << " (in order from vertex 0 to n-1): " << endl;
	map<vertex_desc,size_t>::iterator it;
	for(it = d.begin(); it != d.end(); it++){
		it++;
		if( it == d.end() ){
			it--;
			cout_and_log << (*it).second << endl;
		}
		else{
			it--;
			cout_and_log << (*it).second << ",";
		}
	}
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
				cout_and_log << "Degree sequence does not represent a real graph, fails at d_" << j << ": ";
				for(it = (*it2).begin(); it < (*it2).end()-1; it++){
					cout_and_log << (*it).first << ",";
				}
				cout_and_log << (*it).first << endl;
				cout_and_log << "Press ENTER to continue... " << flush;
				cin.ignore( numeric_limits<streamsize>::max(), '\n' );
				cout_and_log << endl << endl;
				return 0;
			}

			if( j == n && sum % 2 != 0 ){
				cout_and_log << "Degree sequence does not represent a real graph, sum of degrees is not even." << endl;
				cout_and_log << "Press ENTER to continue... " << flush;
				cin.ignore( numeric_limits<streamsize>::max(), '\n' );
				cout_and_log << endl << endl;
				return 0;
			}
		}
	}
	cout_and_log << endl << endl;
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
		cout_and_log << "Cost of anonymizing: " << DA[number_of_vertices - 1] << endl;
		print_degree_sequence("Anonymized degree sequence", DegreeSequence);

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
						for(it = SetOfDegreeSequences[t-1].begin(); it < SetOfDegreeSequences[t-1].end(); it++ ){
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

		cout_and_log << "Cost of anonymizing: " << DA[number_of_vertices - 1] << endl;
		vector<degree_vertex_pair>::iterator it;
		vector< vector<degree_vertex_pair> >::iterator it2;
		for(it2 = SetOfDegreeSequences.back().begin(); it2 < SetOfDegreeSequences.back().end(); it2++ ){
			cout_and_log << "Anonymized degree sequence: ";
			for(it = (*it2).begin(); it < (*it2).end()-1; it++){
				cout_and_log << (*it).first << ",";
			}
			cout_and_log << (*it).first << endl;
		}
		return SetOfDegreeSequences.back();
	}
}

vector< pair<vertex_desc, vertex_desc> > upper_degree_constrained_subgraph(Graph G, vector<degree_vertex_pair> d, map<vertex_desc,size_t> upper_bounds, map<vertex_desc,size_t> delta, size_t number_of_vertices_G_prime, size_t number_of_edges_G_prime, size_t number_of_vertices_G_prime_actual, size_t number_of_edges_G_prime_actual){
	DEBUG_START("Substitute K_d,delta structure for each vertex in G', determine list of edges and initial matching ...");
	cout_and_log << "G' properties:" << endl;
	cout_and_log << "\tNumber of vertices: " << number_of_vertices_G_prime << endl;
	cout_and_log << "\tNumber of edges: " << number_of_edges_G_prime + num_edges(G) << endl; // Not quite right, as we erase edges adjacent to vertices with u_i == 0 in G before this
	cout_and_log << "\tNumber of vertices (actually used): " << number_of_vertices_G_prime_actual << endl;
	cout_and_log << "\tNumber of edges (actually used): " << number_of_edges_G_prime_actual + num_edges(G) << endl;	

	size_t index = 0, edge_index = 0;

	vector< pair<vertex_desc,vertex_desc> > edge_list_G_prime( number_of_edges_G_prime_actual );
	vector<vertex_desc> initial_matching(number_of_vertices_G_prime_actual, graph_traits<Graph>::null_vertex());
	vector< pair< vertex_desc, pair<vertex_desc,vertex_desc> > > external_vertices_pairs; // Maps each new external vertex number to the pair (u,v) = e in G, where e is the edge for which the new vertex was made
	vector<vertex_desc> vertex_K_d_delta;	// Keeps track of which vertex in G correspond to vertex indices in each K_d,delta
	for(size_t i = 0; i < d.size(); i++){
		vertex_desc u = d.at(i).second;
		vector<vertex_desc> internal_vertices, external_vertices;
		adjacency_it ai, ai_end;
		
		// If u_i <= 0 (== 0), then don't add that vertex and it's edges, as it cannot be in the matching anyways
		if( upper_bounds[u] > 0 ){
			for (tie(ai, ai_end) = adjacent_vertices(u, G); ai != ai_end; ++ai){
				external_vertices.push_back(index);
				pair<vertex_desc,vertex_desc> external_pair;
				if( u < *ai ){
					external_pair = make_pair(u, *ai);
				}
				else{
					external_pair = make_pair(*ai, u);
				}
				external_vertices_pairs.push_back( make_pair( index, external_pair ) );
				index++;
				vertex_K_d_delta.push_back( u );
			}
			for(size_t j = 0; j < delta[u]; j++){
				internal_vertices.push_back(index++);
				vertex_K_d_delta.push_back( u );
			}
			vector<vertex_desc>::iterator vit, vit2;
			for(vit = external_vertices.begin(); vit < external_vertices.end(); vit++){
				for(vit2 = internal_vertices.begin(); vit2 < internal_vertices.end(); vit2++){
					edge_list_G_prime[edge_index++] = make_pair(*vit, *vit2);

					// Set up initial matching
					if( initial_matching[*vit] == initial_matching[*vit2] ){ // Only way equality can hold is if initial_matching[*vit] == initial_matching[*vit2] == graph_traits<Graph>::null_vertex()
						initial_matching[*vit] = *vit2;
						initial_matching[*vit2] = *vit;
					}
				}
			}
		}
	}
	DEBUG_END("Substitute K_d,delta structure for each vertex in G', determine list of edges and initial matching ...");

	// Add all edges between the newly created vertices in G' that correspond to each vertex adjacent to another vertex in G
	DEBUG_START("Add edges between external vertices in G' ...");
	vector< pair<vertex_desc, pair<vertex_desc,vertex_desc> > >::iterator it, it2;
	for(it = external_vertices_pairs.begin(); it < external_vertices_pairs.end(); it++){
		for(it2 = it+1; it2 < external_vertices_pairs.end(); it2++){
			if( (*it).second == (*it2).second ){
				edge_list_G_prime.push_back( make_pair( (*it).first, (*it2).first ) ); //add_edge( vertex((*it).first, G_prime), vertex((*it2).first, G_prime), G_prime);
				external_vertices_pairs.erase(it2);
				break;
			}
		}
	}
	DEBUG_END("Add edges between external vertices in G' ...");

	DEBUG_START("Creating graph G' ...");
	Graph G_prime( edge_list_G_prime.begin(), edge_list_G_prime.end(), number_of_vertices_G_prime_actual );
	DEBUG_END("Creating graph G' ...");
	
	// Find maximum cardinality matching in G', which corresponds upper degree constained subgraph G
	//	http://www.boost.org/doc/libs/1_48_0/libs/graph/doc/maximum_matching.html
	DEBUG_START("Finding matching on G' ...");
	bool check_matching = matching<Graph, size_t *, property_map<Graph, vertex_index_t>::type,
		edmonds_augmenting_path_finder, input_initial_matching, maximum_cardinality_matching_verifier>(G_prime, &initial_matching[0], get(vertex_index,G_prime));

	vector< pair<vertex_desc,vertex_desc> > H;
	if( check_matching ){
		cout_and_log << "Maximum cardinality matching size: " << matching_size(G_prime, &initial_matching[0]) << endl;

		// Find edges to be added to G, which corresponds to the maximum cardinality matching in G'
		vertex_iterator ui, ui_end;
		for(tie(ui,ui_end) = vertices(G_prime); ui != ui_end; ++ui){
			if(initial_matching[*ui] != graph_traits<Graph>::null_vertex() && *ui < initial_matching[*ui]){
				size_t index_match_start = vertex_K_d_delta.at(*ui);
				size_t index_match_end = vertex_K_d_delta.at(initial_matching[*ui]);
				if(index_match_start != index_match_end){
					cout_and_log << "Add edge {" << index_match_start << ", " << index_match_end << "} to G to make it k-degree anonymous." << endl;
					H.push_back( make_pair(index_match_start, index_match_end) );
				}
			}
		}
	}
	else{
		cout_and_log << "Maximum cardinality matching failed check." << endl;
	}
	DEBUG_END("Finding matching on G' ...");

	return H;
}

bool augment_matching()
{
	//As an optimization, some of these values can be saved from one
	//iteration to the next instead of being re-initialized each
	//iteration, allowing for "lazy blossom expansion." This is not
	//currently implemented.

	e_size_t timestamp = 0;
	even_edges.clear();

	vertex_iterator_t vi, vi_end;
	for(boost::tie(vi,vi_end) = vertices(g); vi != vi_end; ++vi)
	{
		vertex_descriptor_t u = *vi;

		origin[u] = u;
		pred[u] = u;
		ancestor_of_v[u] = 0;
		ancestor_of_w[u] = 0;
		ds.make_set(u);

		if (mate[u] == graph_traits<Graph>::null_vertex())
		{
			vertex_state[u] = graph::detail::V_EVEN;
			out_edge_iterator_t ei, ei_end;
			for(boost::tie(ei,ei_end) = out_edges(u,g); ei != ei_end; ++ei)
				even_edges.push_back( *ei );
		}
		else
			vertex_state[u] = graph::detail::V_UNREACHED;      
	}

	//end initializations

	vertex_descriptor_t v,w,w_free_ancestor,v_free_ancestor;
	w_free_ancestor = graph_traits<Graph>::null_vertex();
	v_free_ancestor = graph_traits<Graph>::null_vertex(); 
	bool found_alternating_path = false;

	while(!even_edges.empty() && !found_alternating_path)
	{
		// since we push even edges onto the back of the list as
		// they're discovered, taking them off the back will search
		// for augmenting paths depth-first.
		edge_descriptor_t current_edge = even_edges.back();
		even_edges.pop_back();

		v = source(current_edge,g);
		w = target(current_edge,g);

		vertex_descriptor_t v_prime = origin[ds.find_set(v)];
		vertex_descriptor_t w_prime = origin[ds.find_set(w)];

		// because of the way we put all of the edges on the queue,
		// v_prime should be labeled V_EVEN; the following is a
		// little paranoid but it could happen...
		if (vertex_state[v_prime] != graph::detail::V_EVEN)
		{
			std::swap(v_prime,w_prime);
			std::swap(v,w);
		}

		if (vertex_state[w_prime] == graph::detail::V_UNREACHED)
		{
			vertex_state[w_prime] = graph::detail::V_ODD;
			vertex_state[mate[w_prime]] = graph::detail::V_EVEN;
			out_edge_iterator_t ei, ei_end;
			for( boost::tie(ei,ei_end) = out_edges(mate[w_prime], g); ei != ei_end; ++ei)
				even_edges.push_back(*ei);
			pred[w_prime] = v;
		}

		//w_prime == v_prime can happen below if we get an edge that has been
		//shrunk into a blossom
		else if (vertex_state[w_prime] == graph::detail::V_EVEN && w_prime != v_prime) 
		{                                                             
			vertex_descriptor_t w_up = w_prime;
			vertex_descriptor_t v_up = v_prime;
			vertex_descriptor_t nearest_common_ancestor 
				= graph_traits<Graph>::null_vertex();
			w_free_ancestor = graph_traits<Graph>::null_vertex();
			v_free_ancestor = graph_traits<Graph>::null_vertex();

			// We now need to distinguish between the case that
			// w_prime and v_prime share an ancestor under the
			// "parent" relation, in which case we've found a
			// blossom and should shrink it, or the case that
			// w_prime and v_prime both have distinct ancestors that
			// are free, in which case we've found an alternating
			// path between those two ancestors.

			++timestamp;

			while (nearest_common_ancestor == graph_traits<Graph>::null_vertex() && 
				(v_free_ancestor == graph_traits<Graph>::null_vertex() || 
				w_free_ancestor == graph_traits<Graph>::null_vertex()
				)
				)
			{
				ancestor_of_w[w_up] = timestamp;
				ancestor_of_v[v_up] = timestamp;

				if (w_free_ancestor == graph_traits<Graph>::null_vertex())
					w_up = parent(w_up);
				if (v_free_ancestor == graph_traits<Graph>::null_vertex())
					v_up = parent(v_up);

				if (mate[v_up] == graph_traits<Graph>::null_vertex())
					v_free_ancestor = v_up;
				if (mate[w_up] == graph_traits<Graph>::null_vertex())
					w_free_ancestor = w_up;

				if (ancestor_of_w[v_up] == timestamp)
					nearest_common_ancestor = v_up;
				else if (ancestor_of_v[w_up] == timestamp)
					nearest_common_ancestor = w_up;
				else if (v_free_ancestor == w_free_ancestor && 
					v_free_ancestor != graph_traits<Graph>::null_vertex())
					nearest_common_ancestor = v_up;
			}

			if (nearest_common_ancestor == graph_traits<Graph>::null_vertex())
				found_alternating_path = true; //to break out of the loop
			else
			{
				//shrink the blossom
				link_and_set_bridges(w_prime, nearest_common_ancestor, std::make_pair(w,v));
				link_and_set_bridges(v_prime, nearest_common_ancestor, std::make_pair(v,w));
			}
		}      
	}

	if (!found_alternating_path)
		return false;

	// retrieve the augmenting path and put it in aug_path
	reversed_retrieve_augmenting_path(v, v_free_ancestor);
	retrieve_augmenting_path(w, w_free_ancestor);

	// augment the matching along aug_path
	vertex_descriptor_t a,b;
	while (!aug_path.empty())
	{
		a = aug_path.front();
		aug_path.pop_front();
		b = aug_path.front();
		aug_path.pop_front();
		mate[a] = b;
		mate[b] = a;
	}

	return true;

}