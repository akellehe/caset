//
// Created by Andrew Kelleher on 10/19/25.
//

#ifndef CASET_CASET_SRC_VERTEX_H_
#define CASET_CASET_SRC_VERTEX_H_

#include <vector>


class Vertex {

 public:

    Vertex(int id_, const std::vector<double> &coordinates_)
        : id(id_), coordinates(coordinates_) {}

    int getId() const {
        return id;
    }

    const std::vector<double> &getCoordinates() const {
        return coordinates;
    }

 private:
    int id;
    std::vector<double> coordinates;
};

#endif //CASET_CASET_SRC_VERTEX_H_
