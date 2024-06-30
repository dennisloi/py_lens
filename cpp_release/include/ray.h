//
// Created by p4nzer96 on 30/06/24.
//

#ifndef CPP_LENS_RAY_H
#define CPP_LENS_RAY_H

typedef struct {
    float y;
    float x;
}point;


class Ray{
private:
    point endPoint;
    float length;
public:
    float strength;
    std::string color;
    point direction;
    point origin;
};

#endif //CPP_LENS_RAY_H

