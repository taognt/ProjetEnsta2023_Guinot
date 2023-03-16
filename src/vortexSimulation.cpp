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

    
    std::size_t numberOfPoints = cloud.numberOfPoints();
    std::size_t numberOfVortices = vortices.numberOfVortices();
    std::vector<double> buffer_data;// buffer : vector of vortices and then of points from cloud
    int size_of_buffer;
    if(isMobile){
        size_of_buffer = 2*numberOfPoints+3*numberOfVortices;
    }
    else{
        size_of_buffer = 2*numberOfPoints;
    }


    Geometry::Point<double> buffer_pt;
    buffer_data.resize(size_of_buffer);
    double intensity;
    Geometry::Point<double> the_point;
    bool running=true;
    bool START=false;
    

    // Initialize MPI
    MPI_Comm global;
    int rank, nbp;
    MPI_Init(&nargs, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &global);
    MPI_Comm_size(global, &nbp);
    MPI_Comm_rank(global, &rank);
    MPI_Status status;
    std::size_t i; // index of while loops
    std::size_t i_loc; //index of the vortice
    MPI_Request request;



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
            
            
            // on inspecte tous les évènements de la fenêtre qui ont été émis depuis la précédente itération
            sf::Event event;


            // Separer calcul et graphique :
            // Le processus 0 gere la partie graphique et envoi des ordres aux processus s'occupant de la partie calcul
            while (myScreen.pollEvent(event))
            {
                // évènement "fermeture demandée" : on ferme la fenêtre
                if (event.type == sf::Event::Closed){
                    running = false;
                    MPI_Recv(&buffer_data[0], size_of_buffer, MPI_DOUBLE,1, 101, global, MPI_STATUS_IGNORE);
                    MPI_Send(&running, 1, MPI_DOUBLE, 1, 1010, global);
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
                    MPI_Send(&dt, 1, MPI_DOUBLE, 1, 21, global);
                    }
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)){
                    dt /= 2;
                    MPI_Send(&dt, 1, MPI_DOUBLE, 1, 21, global);
                }
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) advance = true;
            }
            
            
            // Attente de message 
            // message
            //MPI_Recv(&buffer_data, size_of_buffer, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, global, &status);
            //std::cout<<"\n\n MESSAGE RECEIVED ! \n"<<std::endl;
            //re-contruction of cloud from buffer_data :
            START = animate | advance;
            if(START){
                buffer_data.clear();

                //buffer_data.clear();

                MPI_Recv(&buffer_data[0], size_of_buffer, MPI_DOUBLE, 1, 101, global, MPI_STATUS_IGNORE);
                if(isMobile){
                    for(std::size_t i=0; i<numberOfVortices; i++){
                        the_point.x = buffer_data[3*i];
                        the_point.y = buffer_data[3*i+1];
                        intensity = buffer_data[3*i+2];
                        vortices.setVortex(i, the_point, intensity);
                    }
                    grid.updateVelocityField(vortices);
                    for(std::size_t i=0; i<numberOfPoints; i++){
                        cloud[i].x = buffer_data[numberOfVortices*3 + 2*i];
                        cloud[i].y = buffer_data[numberOfVortices*3 + 2*i+1];
                    }
                    advance=false;
                }

                else{
                    for(std::size_t i=0; i<numberOfPoints; i++){
                        cloud[i].x = buffer_data[numberOfVortices*3+2*i];
                        cloud[i].y = buffer_data[numberOfVortices*3+ 2*i+1];
                    }
                    advance=false;
                }
            }
            
            
            myScreen.clear(sf::Color::Black);
            std::string strDt = std::string("Time step : ") + std::to_string(dt);
            myScreen.drawText(strDt, Geometry::Point<double>{50, double(myScreen.getGeometry().second-96)});
            myScreen.displayVelocityField(grid, vortices);
            myScreen.displayParticles(grid, vortices, cloud);
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double> diff = end - start;
            std::string str_fps = std::string("FPS : ") + std::to_string(1./diff.count());
            myScreen.drawText(str_fps, Geometry::Point<double>{300, double(myScreen.getGeometry().second-96)});
            myScreen.display();
            
            }



    }


    else if(rank==1){
        while(running){
            MPI_Irecv(&running, 1, MPI_LOGICAL, 0, 1010, global, &request);
            MPI_Irecv(&dt, 1, MPI_DOUBLE, 0, 21, global, &request);

            if (isMobile)
            {
                cloud = Numeric::solve_RK4_movable_vortices(dt, grid, vortices, cloud);

                buffer_data.clear();
                for(std::size_t i=0; i<numberOfVortices; i++){
                    buffer_data.push_back(vortices.getCenter(i).x);
                    buffer_data.push_back(vortices.getCenter(i).y);
                    buffer_data.push_back(vortices.getIntensity(i));
                }
                for(std::size_t i=0; i<numberOfPoints; i++){
                    buffer_data.push_back(cloud[i].x);
                    buffer_data.push_back(cloud[i].y);
                }

                MPI_Send(&buffer_data[0], 2*numberOfPoints+3*numberOfVortices, MPI_DOUBLE, 0, 101, global);
            }   

            else
            {
                cloud = Numeric::solve_RK4_fixed_vortices(dt, grid, cloud);
                buffer_data.clear();
                for(std::size_t i=0; i<numberOfPoints; i++){
                    buffer_data.push_back(cloud[i].x);
                    buffer_data.push_back(cloud[i].y);
                }
                MPI_Send(&buffer_data[0], 2*numberOfPoints, MPI_DOUBLE, 0, 101, global);
            }
    
        }
        
    }

    

    // Finalize MPI
    MPI_Finalize();
    return EXIT_SUCCESS;
 }