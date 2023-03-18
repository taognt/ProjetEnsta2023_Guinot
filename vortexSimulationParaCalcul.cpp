#include <SFML/Window/Keyboard.hpp>
#include <ios>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <chrono>
#include "cartesian_grid_of_speed.hpp"
#include "vortex.hpp"
#include "cloud_of_points.hpp"
#include "runge_kutta.hpp"
#include "screen.hpp"

#include <mpi.h>

// mpirun -np <nb process> ./vortexSimulation.exe <data> 1280 1024

// si on veut deux exe:
// mpirun -np 1 ./affiche.exe : -np 7 ./calcul.exe

auto readConfigFile( std::ifstream& input )
{
    using point=Simulation::Vortices::point;

    int isMobile;
    std::size_t nbVortices;
    Numeric::CartesianGridOfSpeed cartesianGrid;
    Geometry::CloudOfPoints cloudOfPoints;
    constexpr std::size_t maxBuffer = 8192;
    char buffer[maxBuffer];
    std::string sbuffer;
    std::stringstream ibuffer;
    // Lit la première ligne de commentaire :
    input.getline(buffer, maxBuffer); // Relit un commentaire
    input.getline(buffer, maxBuffer);// Lecture de la grille cartésienne
    sbuffer = std::string(buffer,maxBuffer);
    ibuffer = std::stringstream(sbuffer);
    double xleft, ybot, h;
    std::size_t nx, ny;
    ibuffer >> xleft >> ybot >> nx >> ny >> h;
    cartesianGrid = Numeric::CartesianGridOfSpeed({nx,ny}, point{xleft,ybot}, h);
    input.getline(buffer, maxBuffer); // Relit un commentaire
    input.getline(buffer, maxBuffer); // Lit mode de génération des particules
    sbuffer = std::string(buffer,maxBuffer);
    ibuffer = std::stringstream(sbuffer);
    int modeGeneration;
    ibuffer >> modeGeneration;
    if (modeGeneration == 0) // Génération sur toute la grille 
    {
        std::size_t nbPoints;
        ibuffer >> nbPoints;
        cloudOfPoints = Geometry::generatePointsIn(nbPoints, {cartesianGrid.getLeftBottomVertex(), cartesianGrid.getRightTopVertex()});
    }
    else 
    {
        std::size_t nbPoints;
        double xl, xr, yb, yt;
        ibuffer >> xl >> yb >> xr >> yt >> nbPoints;
        cloudOfPoints = Geometry::generatePointsIn(nbPoints, {point{xl,yb}, point{xr,yt}});
    }
    // Lit le nombre de vortex :
    input.getline(buffer, maxBuffer); // Relit un commentaire
    input.getline(buffer, maxBuffer); // Lit le nombre de vortex
    sbuffer = std::string(buffer, maxBuffer);
    ibuffer = std::stringstream(sbuffer);
    try {
        ibuffer >> nbVortices;        
    } catch(std::ios_base::failure& err)
    {
        std::cout << "Error " << err.what() << " found" << std::endl;
        std::cout << "Read line : " << sbuffer << std::endl;
        throw err;
    }
    Simulation::Vortices vortices(nbVortices, {cartesianGrid.getLeftBottomVertex(),
                                               cartesianGrid.getRightTopVertex()});
    input.getline(buffer, maxBuffer);// Relit un commentaire
    for (std::size_t iVortex=0; iVortex<nbVortices; ++iVortex)
    {
        input.getline(buffer, maxBuffer);
        double x,y,force;
        std::string sbuffer(buffer, maxBuffer);
        std::stringstream ibuffer(sbuffer);
        ibuffer >> x >> y >> force;
        vortices.setVortex(iVortex, point{x,y}, force);
    }
    input.getline(buffer, maxBuffer);// Relit un commentaire
    input.getline(buffer, maxBuffer);// Lit le mode de déplacement des vortex
    sbuffer = std::string(buffer,maxBuffer);
    ibuffer = std::stringstream(sbuffer);
    ibuffer >> isMobile;
    return std::make_tuple(vortices, isMobile, cartesianGrid, cloudOfPoints);
}


int main( int nargs, char* argv[] )
{   


    char const* filename;
    if (nargs==1)
    {
        std::cout << "Usage : vortexsimulator <nom fichier configuration>" << std::endl;
        return EXIT_FAILURE;
    }

    filename = argv[1];
    std::ifstream fich(filename);
    auto config = readConfigFile(fich);
    fich.close();

    std::size_t resx=800, resy=600;
    if (nargs>3)
    {
        resx = std::stoull(argv[2]);
        resy = std::stoull(argv[3]);
    }

    auto vortices = std::get<0>(config);
    auto isMobile = std::get<1>(config);
    auto grid     = std::get<2>(config);
    auto cloud    = std::get<3>(config);

    
    grid.updateVelocityField(vortices);


    bool animate=false;
    double dt = 0.1;


    // Initialize MPI
    MPI_Comm global;
    int rank, nbp;
    MPI_Init(&nargs, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &global);
    MPI_Comm_size(global, &nbp);
    MPI_Comm_rank(global, &rank);
    MPI_Request request;

    
    // variables used 
    std::size_t numberOfPoints = cloud.numberOfPoints()/(nbp-1);
    std::size_t numberOfVortices = vortices.numberOfVortices();
    std::vector<double> buffer_data;// buffer : vector of vortices and then of points from cloud
    std::vector<double> buffer_data_last;
    int size_of_buffer;
    int size_of_last_buffer;
    size_of_buffer = 2*(numberOfPoints/(nbp-1));
    buffer_data.resize(size_of_buffer);
    buffer_data_last.resize(size_of_last_buffer);
    double intensity;
    Geometry::Point<double> the_point;
    bool running=true; //The window is open
    bool START=false; // The calculus is asked / needed
    




    // Process 0 deal with the displaying part
    if(rank==0){
        std::cout << "######## Vortex simultor ########" << std::endl << std::endl;
        std::cout << "Press P for play animation " << std::endl;
        std::cout << "Press S to stop animation" << std::endl;
        std::cout << "Press right cursor to advance step by step in time" << std::endl;
        std::cout << "Press down cursor to halve the time step" << std::endl;
        std::cout << "Press up cursor to double the time step" << std::endl;
        std::cout << "number of points : " << numberOfPoints << std::endl;
        std::cout << "number of vortices : " << numberOfVortices << std::endl;
        
        Graphisme::Screen myScreen( {resx,resy}, {grid.getLeftBottomVertex(), grid.getRightTopVertex()} );
        while (myScreen.isOpen()){
            bool advance = false;
            auto start = std::chrono::system_clock::now();
            
            sf::Event event;
            while (myScreen.pollEvent(event))
            {
                // évènement "fermeture demandée" : on ferme la fenêtre
                if (event.type == sf::Event::Closed){
                    running = false;
                    //MPI_Recv(&buffer_data[0], size_of_buffer, MPI_DOUBLE,1, 101, global, MPI_STATUS_IGNORE);
                    for(int k=1; k<nbp; ++k){
                        MPI_Send(&running, k, MPI_DOUBLE, 1, 1010, global);
                    }
                    myScreen.close();
                }

                if (event.type == sf::Event::Resized)
                {
                    // on met à jour la vue, avec la nouvelle taille de la fenêtre
                    myScreen.resize(event);
                }
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::P)) animate = true;
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) animate = false;
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)){
                    dt *= 2;
                    for(int k=1; k<nbp; ++k){
                        MPI_Send(&dt, 1, MPI_DOUBLE, k, 21, global);
                    }
                    }
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)){
                    dt /= 2;
                    for(int k=1; k<nbp; ++k){
                        MPI_Send(&dt, 1, MPI_DOUBLE, k, 21, global);
                    }
                }
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) advance = true;
            }
            
            // Affichage et evaluation 

            myScreen.clear(sf::Color::Black);
            std::string strDt = std::string("Time step : ") + std::to_string(dt);
            myScreen.drawText(strDt, Geometry::Point<double>{50, double(myScreen.getGeometry().second-96)});
            myScreen.displayVelocityField(grid, vortices);
            myScreen.displayParticles(grid, vortices, cloud);
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double> diff = end - start;
            //std::cout<<"Temps : " << diff.count()<<std::endl;
            std::string str_fps = std::string("FPS : ") + std::to_string(1./diff.count());
            myScreen.drawText(str_fps, Geometry::Point<double>{300, double(myScreen.getGeometry().second-96)});
            myScreen.display();
            
            
            // Update du cloud et de vortices :
            START = animate | advance;
            if(START){
                buffer_data.clear();

                // the first processes
                for(int id=0; id<nbp-1; ++id){
                    MPI_Recv(&buffer_data[0], size_of_buffer, MPI_DOUBLE, id, 101, global, MPI_STATUS_IGNORE);

                    for(std::size_t i=0; i<size_of_buffer*2; i++){
                        int i_loc = i + (id-1)*size_of_buffer;
                        cloud[i_loc].x = buffer_data[2*i];
                        cloud[i_loc].y = buffer_data[2*i+1];
                    }
                    //advance=false;

                }

                // Last process
                MPI_Recv(&buffer_data_last[0], size_of_last_buffer, MPI_DOUBLE, nbp-1, 101, global, MPI_STATUS_IGNORE);
                for(std::size_t i=0; i<size_of_buffer*2; i++){
                    int i_loc = i + (nbp-2)*size_of_buffer;
                    cloud[i_loc].x = buffer_data[2*i];
                    cloud[i_loc].y = buffer_data[2*i+1];
                }

            }
            

            }



    }

    else if(rank!=nbp-1){
        Geometry::CloudOfPoints cloud_calcul(size_of_buffer);
        for(int iPoint = 0; iPoint<size_of_buffer; ++iPoint){
            cloud_calcul[iPoint] = cloud[(rank-1)*size_of_buffer + iPoint];
        }

        while(running){
            MPI_Irecv(&running, 1, MPI_LOGICAL, 0, 1010, global, &request);
            MPI_Irecv(&dt, 1, MPI_DOUBLE, 0, 21, global, &request);


            cloud_calcul = Numeric::solve_RK4_fixed_vortices(dt, grid, cloud_calcul);
            buffer_data.clear();
            for(std::size_t i=0; i<cloud_calcul.numberOfPoints(); i++){
                buffer_data.push_back(cloud_calcul[i].x);
                buffer_data.push_back(cloud_calcul[i].y);
            }
            MPI_Send(&buffer_data[0], size_of_buffer, MPI_DOUBLE, 0, 101, global);
    
        }
        
    }
    // rank == nbp-1
    else{
        std::cout<<"rank : "<< rank<<std::endl;
        Geometry::CloudOfPoints cloud_calcul(size_of_last_buffer);
        for(int iPoint = 0; iPoint<size_of_last_buffer; ++iPoint){
            cloud_calcul[iPoint] = cloud[(rank-1)*size_of_last_buffer + iPoint];
        }

        while(running){
            MPI_Irecv(&running, 1, MPI_LOGICAL, 0, 1010, global, &request);
            MPI_Irecv(&dt, 1, MPI_DOUBLE, 0, 21, global, &request);


            cloud_calcul = Numeric::solve_RK4_fixed_vortices(dt, grid, cloud_calcul);
            buffer_data_last.clear();
            for(std::size_t i=0; i<cloud_calcul.numberOfPoints(); i++){
                buffer_data_last.push_back(cloud_calcul[i].x);
                buffer_data_last.push_back(cloud_calcul[i].y);
            }
            MPI_Send(&buffer_data_last[0], size_of_last_buffer, MPI_DOUBLE, 0, 101, global);
    
        }


    }

    // Finalize MPI
    MPI_Finalize();
    return EXIT_SUCCESS;
 }