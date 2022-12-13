#include "Descriptor.hpp"
#include <iostream>
#include <iterator>
#include <fstream>
#include <string>
#include <vector>



int main(int argc, char *argv[])
{
    
    std:: string D_str = argv[1];
    
    //sift
    //std::vector<aliceVision::feature::Descriptor<unsigned char, 128>> D;
    //std::vector<aliceVision::feature::Descriptor<float, 128>> D;
    //surf
    //std::vector<aliceVision::feature::Descriptor<float, 64>> D;
    //akaze
    std::vector<aliceVision::feature::Descriptor<float, 64>> D;
    //std::vector<aliceVision::feature::Descriptor<unsigned char, 144>> D;
       
    
    aliceVision::feature::loadDescsFromBinFile(D_str, D);
    
    std::cout << "descriptor size " << D[0].size() << "\n";
    std::cout << "Number of features " << D.size() << "\n";
    //std::cout << D[0];

    //Writting descriptor as a txt file
    
    std::string file_name = D_str+".txt";
    std::cout << "Saved in :" + file_name + "\n";
    
    std::ofstream MyFile(file_name);
    for (const auto &e : D) MyFile << e << "\n";

    return 0;
}
