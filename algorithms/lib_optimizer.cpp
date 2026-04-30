#include <iostream>
#include <vector>
#include <algorithm>

/**
 * CareerPath Optimizer - C++ Engine
 * =================================
 * High-performance implementation of Greedy and DP algorithms
 * for course optimization.
 */

struct Course {
    int id;
    int hours;
    int value;
    double ratio;
};

// Sort helper for Greedy
bool compareCourses(const Course& a, const Course& b) {
    return a.ratio > b.ratio;
}

extern "C" {
    /**
     * Greedy Algorithm
     * Returns the number of selected courses.
     * result_indices will be filled with the indices of selected courses.
     */
    int greedy_optimize(int* hours, int* values, int n, int max_hours, int* result_indices) {
        std::vector<Course> courses;
        for (int i = 0; i < n; i++) {
            courses.push_back({i, hours[i], values[i], (double)values[i] / hours[i]});
        }

        std::sort(courses.begin(), courses.end(), compareCourses);

        int current_hours = 0;
        int selected_count = 0;

        for (int i = 0; i < n; i++) {
            if (current_hours + courses[i].hours <= max_hours) {
                current_hours += courses[i].hours;
                result_indices[selected_count++] = courses[i].id;
            }
        }

        return selected_count;
    }

    /**
     * 0/1 Knapsack Dynamic Programming
     * Returns the number of selected courses.
     * result_indices will be filled with the indices of selected courses.
     */
    int dp_optimize(int* hours, int* values, int n, int max_hours, int* result_indices) {
        if (n == 0 || max_hours <= 0) return 0;

        // Using a 1D DP array optimization to save space, 
        // but we need a 2D table to backtrack selected items.
        // For educational purposes and backtracking, 2D is clearer.
        std::vector<std::vector<int>> dp(n + 1, std::vector<int>(max_hours + 1, 0));

        for (int i = 1; i <= n; i++) {
            int h = hours[i - 1];
            int v = values[i - 1];
            for (int w = 0; w <= max_hours; w++) {
                if (h <= w) {
                    dp[i][w] = std::max(dp[i - 1][w], v + dp[i - 1][w - h]);
                } else {
                    dp[i][w] = dp[i - 1][w];
                }
            }
        }

        // Backtracking
        int w = max_hours;
        int selected_count = 0;
        std::vector<int> temp_indices;

        for (int i = n; i > 0; i--) {
            if (dp[i][w] != dp[i - 1][w]) {
                temp_indices.push_back(i - 1);
                w -= hours[i - 1];
            }
        }

        // Fill result_indices in original order
        for (int i = temp_indices.size() - 1; i >= 0; i--) {
            result_indices[selected_count++] = temp_indices[i];
        }

        return selected_count;
    }
}
