#include <stdlib.h>
#include <stdio.h>
#include "physics/entity.c"

typedef struct{
    entity* predators;
    entity* preys;
    int* alive;
    entity* obstacles;
    
    double x_limit;
    double y_limit;
    
    int num_preds;
    int num_preys;
    int num_obstacles;
    
    double r_obst_ub; // upper bound for obstacle radius
    double r_obst_lb; // lower bound for obstacle radius    
    double prey_radius;
    double pred_radius;
    
    double pred_speed;
    double prey_speed;
    double w_timestep; // world timestep
    
    int* prey_order;
    int* pred_order;
} FGame;


double double_rand(){
    return (double)random() / RAND_MAX;
}

void shuffle_array(int* a, int n){
    
    for(int i=n-1; i>0; i--){
        int j = rand() % (i + 1);
      
        int temp = a[i];
        a[i] = a[j];
        a[j] = temp;
    }
}

void seed(int n){
    srand(n);
}


FGame* game_init(double xl, double yl,
                 int n_preds, int n_preys, int n_obsts,
                 double r_ub, double r_lb, double prey_r, double pred_r,
                 double pred_s, double prey_s,
                 double wt){
                 
    FGame* F = (FGame*) malloc(sizeof(FGame));
    
    F -> predators = (entity* )malloc(sizeof(entity) * n_preds);
    F -> preys = (entity* )malloc(sizeof(entity) * n_preys);
    F -> alive = (int* )malloc(sizeof(int) * n_preys);
    F -> obstacles = (entity* )malloc(sizeof(entity) * n_obsts);
    
    F -> x_limit = xl;
    F -> y_limit = yl;
    
    F -> num_preds = n_preds;
    F -> num_preys = n_preys;
    F -> num_obstacles = n_obsts;
    
    F -> r_obst_ub = r_ub;
    F -> r_obst_lb = r_lb;
    F -> prey_radius = prey_r;
    F -> pred_radius = pred_r;
    
    F -> pred_speed = pred_s;
    F -> prey_speed = prey_s;
    
    F -> w_timestep = wt;
    
    F -> prey_order = (int* ) malloc(sizeof(int) * n_preys);
    for(int i=0; i<n_preys; i++)
        F -> prey_order[i] = i;
        
    F -> pred_order = (int* ) malloc(sizeof(int) * n_preds);
    for(int i=0; i<n_preds; i++)
        F -> pred_order[i] = i;
    
    return F;
}


entity get_prey(FGame* F, int i){
    return F -> preys[i];
}


int get_alive(FGame* F, int i){
    return F -> alive[i];
}


entity get_predator(FGame* F, int i){
    return F -> predators[i];
}


entity get_obstacle(FGame* F, int i){
    return F -> obstacles[i];
}


void step(FGame* F, double* action_preys, double* action_predators){
    
    FGame G = *F;
    
    for(int i=0; i<G.num_preys; i++){
        if (G.alive[i])
            move(&G.preys[i], action_preys[i], G.w_timestep);
    }
    
    for(int i=0; i<G.num_preds; i++)
        move(&G.predators[i], action_predators[i], G.w_timestep);
        
    int corrected = 1;
    int it_num = 0;
    int shuffle_count = 0;
    
    while (corrected){
        corrected = 0;
        for(int k=0; k<G.num_preys; k++){
            int i = G.prey_order[k];
            int this_corrected = 0;
            force_clip_position(&G.preys[i], -G.x_limit, -G.y_limit, G.x_limit, G.y_limit);
            for(int j=0; j<G.num_obstacles; j++)
                this_corrected += force_not_intersect(&G.preys[i], &G.obstacles[j]);
        
            if (!this_corrected){
                for(int t=0; t<G.num_preys; t++){
                    int j = G.prey_order[t];
                    if (i==j)
                        continue;
                    this_corrected += force_not_intersect(&G.preys[i], &G.preys[j]);
                }
            }
            if(!this_corrected){
                for(int j=0; j<G.num_obstacles; j++)
                    this_corrected += force_not_intersect(&G.preys[i], &G.obstacles[j]);
            }
            corrected += this_corrected;
            force_clip_position(&G.preys[i], -G.x_limit, -G.y_limit, G.x_limit, G.y_limit);
        }
        
        if (!corrected)
            break;
            
        if (it_num > G.num_preys * G.num_preys){
            it_num = 0;
            shuffle_array(G.prey_order, G.num_preys);
            shuffle_count += 1;
        }
        
        if (shuffle_count > G.num_preys * G.num_preys * 3)
           corrected = 0;
           
        it_num += 1;
    }
    
    corrected = 1;
    it_num = 0;
    shuffle_count = 0;
    while (corrected){
        corrected = 0;
        for(int k=0; k<G.num_preds; k++){
            int i = G.pred_order[k];
            force_clip_position(&G.predators[i], -G.x_limit, -G.y_limit, G.x_limit, G.y_limit);
            for(int j=0; j<G.num_obstacles; j++)
                this_corrected += force_not_intersect(&G.predators[i], &G.obstacles[j]);
        
            if (!this_corrected){
                for(int t=0; t<G.num_preds; t++){
                    int j = G.pred_order[t];
                    if (i==j)
                        continue;
                    this_corrected += force_not_intersect(&G.predators[i], &G.predators[j]);
                }
            }
            if(!this_corrected){
                for(int j=0; j<G.num_obstacles; j++)
                    this_corrected += force_not_intersect(&G.predators[i], &G.obstacles[j]);
            }
            corrected += this_corrected;
            force_clip_position(&G.predators[i], -G.x_limit, -G.y_limit, G.x_limit, G.y_limit);
        }
 
        if (!corrected)
            break;
 
        if (it_num > G.num_preds * G.num_preds){
            it_num = 0;
            shuffle_array(G.pred_order, G.num_preds);
            shuffle_count += 1;
        }
 
        if (shuffle_count > G.num_preds * G.num_preds * 3)
           corrected = 0;
 
        it_num += 1;
    }
    
    for(int i=0; i<G.num_preys; i++){
        for(int j=0; j<G.num_preds; j++){
            if (is_intersect(&G.preys[i], &G.predators[j]))
                G.alive[i] = 0;
        }
    }
}

void reset(FGame* F){
    
    free(F -> obstacles);
    F -> obstacles = (entity*) malloc(sizeof(entity) * (F -> num_obstacles));
    
    free(F -> preys);
    free(F -> alive);
    F -> preys = (entity*) malloc(sizeof(entity) * (F -> num_preys));
    F -> alive = (int*) malloc(sizeof(int) * (F -> num_preys));
    
    free(F -> predators);
    F -> predators = (entity*) malloc(sizeof(entity) * (F -> num_preds));
    
    FGame G = *F;
    
    for(int i=0; i<G.num_obstacles; i++){
        double r = double_rand() * (G.r_obst_ub - G.r_obst_lb) + G.r_obst_lb;
        double x = (2 * double_rand() - 1) * (G.x_limit - r);
        double y = (2 * double_rand() - 1) * (G.y_limit - r);
        entity* e = Entity_init(r, 0., x, y);
        G.obstacles[i] = *e;
        free(e);
    }
    
    for(int i=0; i<G.num_preys; i++){
        int created = 0;
        while (!created){
            created = 1;
            double x = (2 * double_rand() - 1) * (G.x_limit - G.prey_radius);
            double y = (2 * double_rand() - 1) * (G.y_limit - G.prey_radius);
            entity* e = Entity_init(G.prey_radius, G.prey_speed, x, y);
            for(int j=0; j<G.num_obstacles; j++){
                if (is_intersect(&(G.obstacles[j]), e)){
                    created = 0;
                    free(e);
                    break;
                }
            }
            if (created){
                for(int j=0; j<i; j++){
                    if (is_intersect(&(G.preys[j]), e)){
                        created = 0;
                        free(e);
                        break;
                    }
                }
            }
            if (created){
                G.preys[i] = *e;
                G.alive[i] = 1;
                free(e);
            } 
        }
    }
    
    
    for(int i=0; i<G.num_preds; i++){
        int created = 0;
        while (!created){
            created = 1;
            double x = (2 * double_rand() - 1) * (G.x_limit - G.pred_radius);
            double y = (2 * double_rand() - 1) * (G.y_limit - G.pred_radius);
            entity* e = Entity_init(G.pred_radius, G.pred_speed, x, y);
            for(int j=0; j<G.num_obstacles; j++){
                if (is_intersect(&(G.obstacles[j]), e)){
                    created = 0;
                    free(e);
                    break;
                }
            }
            if (created){
                for(int j=0; j<G.num_preys; j++){
                    if (is_intersect(&(G.preys[j]), e)){
                        created = 0;
                        free(e);
                        break;
                    }
                }
            }
            if (created){
                for(int j=0; j<i; j++){
                    if (is_intersect(&(G.predators[j]), e)){
                        created = 0;
                        free(e);
                        break;
                    }
                }
            }
            if (created){
                G.predators[i] = *e;
                free(e);
            }
        }
    }
}
