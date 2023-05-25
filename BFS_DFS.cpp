#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>

using namespace std;

class Graph {
private:
    int numVertices;
    vector<vector<int>> adjacencyList;

public:
    Graph(int numVertices) : numVertices(numVertices) {
        adjacencyList.resize(numVertices);
    }

    void addEdge(int src, int dest) {
        adjacencyList[src].push_back(dest);
        adjacencyList[dest].push_back(src);
    }

    void BFS(int startVertex) {
        vector<bool> visited(numVertices, false);
        queue<int> queue;

        visited[startVertex] = true;
        queue.push(startVertex);

        while (!queue.empty()) {
            int currentVertex = queue.front();
            queue.pop();
            cout << currentVertex << " ";

            #pragma omp parallel for
            for (int neighbor : adjacencyList[currentVertex]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    queue.push(neighbor);
                }
            }
        }
    }

    void DFS(int startVertex) {
        vector<bool> visited(numVertices, false);
        stack<int> stack;

        stack.push(startVertex);

        while (!stack.empty()) {
            int currentVertex = stack.top();
            stack.pop();

            if (!visited[currentVertex]) {
                visited[currentVertex] = true;
                cout << currentVertex << " ";

                #pragma omp parallel for
                for (int neighbor : adjacencyList[currentVertex]) {
                    stack.push(neighbor);
                }
            }
        }
    }
};

int main() {
    int numVertices = 6;
    Graph graph(numVertices);

    // Add edges to the graph
    graph.addEdge(0, 1);
    graph.addEdge(0, 2);
    graph.addEdge(1, 3);
    graph.addEdge(2, 3);
    graph.addEdge(2, 4);
    graph.addEdge(3, 4);
    graph.addEdge(3, 5);

    cout << "BFS Traversal: ";
    graph.BFS(0);
    cout << endl;

    cout << "DFS Traversal: ";
    graph.DFS(0);
    cout << endl;

    return 0;
}
