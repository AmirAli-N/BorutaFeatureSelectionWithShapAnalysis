setwd("//ahmct-065/teams/PMRF/Amir/")

library(data.table)
library(dplyr)
library(tidyr)
library(caret)
library(anytime)
library(e1071)
library(DMwR)
library(glmnet)
library(doParallel)

library(xgboost)
library(DiagrammeR)
library(ggplot2)
library(hrbrthemes)
library(viridis)
library(ggrepel)
library(SHAPforxgboost)
library(forcats)
library(Boruta)
library(reshape2)


library(Ckmeans.1d.dp)
library(devtools)
library(xgboostExplainer)


set.seed(123)

df=fread(file="./bin/LEMO_CHP.by.roadCond_workOrderDate.csv", sep=",", header=TRUE)
df[df==""]=NA

#select features
colnames(df)
selected_cols=c("work_date", "activity", "district", "county", "route", "work_duration", "work_length", 
                "closure_id", "closure_coverage", "closure_length", "closure_workType", "closure_duration", "closure_cozeepMazeep", 
                "closure_detour", "closure_type", "closure_facility", "closure_lanes",
                "surface_type", "num_lanes", "road_use", "road_width", "median_type", "barrier_type", "hwy_group", "access_type", 
                "terrain_type", "road_speed", "road_adt", "population_code", "peak_aadt", "aadt", "truck_aadt", "collision_density11_12", "collision_id", 
                "collision_time", "collision_day", "collision_weather_cond_1", "collision_weather_cond_2", "collision_location_type", 
                "collision_ramp_intersection", "collision_severity", "collision_num_killed", "collision_num_injured", "collision_party_count", 
                "collision_prime_factor", "collision_violation_cat", "collision_surface_cond", "collision_road_cond_1", "collision_road_cond_2", 
                "collision_lighting_cond", "collision_control_device", "collision_road_type")

#clean up the selected features and convert to type
source("./Codes/FUNC(Dataset CleanUp).R")
df=cleanUp_Dataset(df, selected_cols)

#check clean up process
df %>% str

#filter rows for a complete data set, in that, no features except collision and closure features should be missing
df=na.omit(setDT(df), cols = c("work_month", "work_day", "district", "county", "route", "activity", "work_duration", "work_length", 
                               "surface_type", "num_lanes", "road_use", "road_width", "median_type", "barrier_type", "hwy_group", 
                               "access_type", "terrain_type", "road_speed", "road_adt", "population_code", 
                               "peak_aadt", "aadt", "truck_aadt", "collision_density11_12"))

#create training and testing data set
train.ind=createDataPartition(df$collision_id, times = 1, p=0.7, list = FALSE)
training.df=df[train.ind, ]
testing.df=df[-train.ind, ]

#check and plot response variable class
length(which(training.df$collision_id==1))/length(training.df$collision_id)
ggplot(data=training.df, aes(x=collision_id, fill=collision_id))+
  geom_bar()+
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5, size=14),
        axis.title.x = element_text(size = 20, face="bold"),
        axis.text.y = element_text(size=14),
        axis.title.y = element_text(size=20, face = "bold"), legend.position = "none")+
  ylab("Count")+
  xlab("Class collision")
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
#drop collision and closure columns, some of NA variabels can be translated to 0-1 categories or numerics
train.df=training.df[,-c("closure_workType", "closure_duration", "closure_type", "closure_facility")]
train.df$closure_cozeepMazeep=ifelse(is.na(train.df$closure_cozeepMazeep), 0, 1)
train.df$closure_detour=ifelse(is.na(train.df$closure_detour), 0, 1)

train.df=train.df[,-c("closure_lanes")]
train.df$closure_coverage[is.na(train.df$closure_coverage)]=0
train.df$closure_coverage=abs(train.df$closure_coverage)
train.df$closure_length[is.na(train.df$closure_length)]=0

train.df=train.df[,-c("collision_time", "collision_day", "collision_weather_cond_1", "collision_weather_cond_2", 
                                    "collision_location_type", "collision_ramp_intersection", "collision_severity", "collision_prime_factor", 
                                    "collision_violation_cat", "collision_surface_cond", "collision_road_cond_1", "collision_road_cond_2", 
                                    "collision_lighting_cond", "collision_control_device", "collision_road_type")]

train.df=train.df[,-c("collision_num_killed", "collision_num_injured", "collision_party_count")]

test.df=testing.df[,-c("closure_workType", "closure_duration", "closure_type", "closure_facility")]
test.df$closure_cozeepMazeep=ifelse(is.na(test.df$closure_cozeepMazeep), 0, 1)
test.df$closure_detour=ifelse(is.na(test.df$closure_detour), 0, 1)

test.df=test.df[,-c("closure_lanes")]
test.df$closure_coverage[is.na(test.df$closure_coverage)]=0
test.df$closure_coverage=abs(test.df$closure_coverage)
test.df$closure_length[is.na(test.df$closure_length)]=0

test.df=test.df[,-c("collision_time", "collision_day", "collision_weather_cond_1", "collision_weather_cond_2", 
                    "collision_location_type", "collision_ramp_intersection", "collision_severity", "collision_prime_factor", 
                    "collision_violation_cat", "collision_surface_cond", "collision_road_cond_1", "collision_road_cond_2", 
                    "collision_lighting_cond", "collision_control_device", "collision_road_type")]

test.df=test.df[,-c("collision_num_killed", "collision_num_injured", "collision_party_count")]
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
#prepare sparse matrices for xgboost
#preprocess.mod=preProcess(train.df, method = c("center", "scale"), rangeBounds = c(0,1))
`isnot.factor` = Negate(`is.factor`)
train.num.df=train.df%>%select_if(isnot.factor)
train.num.df=apply(train.num.df, 2, function(col) reshape::rescaler(col, type="range"))
train.df[, which(colnames(train.df)%in%colnames(train.num.df))]=train.num.df
rm(train.num.df)

test.num.df=test.df%>%select_if(isnot.factor)
test.num.df=apply(test.num.df, 2, function(col) reshape::rescaler(col, type="range"))
test.df[, which(colnames(test.df)%in%colnames(test.num.df))]=test.num.df
rm(test.num.df)

dtrain=sparse.model.matrix(collision_id~.-1, train.df)
label=as.numeric(as.character(train.df$collision_id))
dtest=sparse.model.matrix(collision_id~.-1, test.df)

#evaluate the weight of each class in response variable
sumwpos=sum(label==1)
sumwneg=sum(label==0)

##############################################################
#########################################parameter hypertuning
#register parallel backend
myCl=makeCluster(detectCores()-1, outfile="Log.txt")
registerDoParallel(myCl)

#define the parameter grid
xgb.grid=expand.grid(nrounds=100, 
                     eta=seq(0.1, 1, 0.1),
                     max_depth=seq(3, 10, 1),
                     gamma = seq(0, 10, 2.5), 
                     subsample = 0.7,
                     min_child_weight = c(1, 3, 5), 
                     colsample_bytree = 1)

#define caret training controls
xgb.control=trainControl(method = "cv",
                         number = 5,
                         verboseIter = TRUE,
                         returnData = FALSE,
                         returnResamp = "none",
                         classProbs = TRUE,
                         allowParallel = TRUE)

xgb.train = train(x = dtrain,
                  y = factor(label, 
                             labels = c("No.Collision", "Collision")),
                  trControl = xgb.control,
                  tuneGrid = xgb.grid,
                  method = "xgbTree",
                  scale_pos_weight=sumwneg/sumwpos,
                  tree_method="hist")

#result of the grid search
xgb.train$bestTune
params=list("eta"=xgb.train$bestTune$eta,
            "max_depth"=xgb.train$bestTune$max_depth,
            "gamma"=xgb.train$bestTune$gamma,
            "min_child_weight"=xgb.train$bestTune$min_child_weight,
            "nthread"=4,
            "objective"="binary:logistic")

#run a cross-validated search for the best number of iterations
xgb.crv=xgb.cv(params = params,
               data = dtrain,
               nrounds = 500,
               nfold = 5,
               label = label,
               showsd = TRUE,
               metrics = list("auc", "rmse", "logloss"),
               stratified = TRUE,
               verbose = TRUE,
               print_every_n = 1L,
               early_stopping_rounds = 50,
               scale_pos_weight=sumwneg/sumwpos)

#result of the cross-validation
xgb.crv$best_iteration
#plot of the cross-validated training and test error
ggplot(xgb.crv$evaluation_log, aes(x=iter))+
  geom_line(aes(y=train_auc_mean, 
                color="Training accuracy"), 
            size=1.2)+
  geom_ribbon(aes(y=train_auc_mean, 
                  ymax=train_auc_mean+train_auc_std,
                  ymin=train_auc_mean-train_auc_std,
                  alpha=0.3))+
  geom_line(aes(y=test_auc_mean, 
                color="Testing accuracy"), 
            size=1.2)+
  geom_ribbon(aes(y=train_auc_mean, 
                  ymax=test_auc_mean+test_auc_std,
                  ymin=test_auc_mean-test_auc_std,
                  alpha=0.3))+
  theme_ipsum(axis_title_just = "center")+
  theme(plot.title = element_blank(),
        legend.text = element_text(size = 18, 
                                   family = "Century Gothic", 
                                   color = "black"),
        axis.text.x = element_text(hjust = 0.5, 
                                   size=18, 
                                   family = "Century Gothic", 
                                   color = "black"),
        axis.title.x = element_text(size = 18, 
                                    family = "Century Gothic", 
                                    color = "black", 
                                    margin = margin(15, 0, 0, 0)),
        axis.text.y = element_text(size=18, 
                                   family = "Century Gothic", 
                                   color = "black"),
        axis.title.y = element_text(size=18, 
                                    family = "Century Gothic", 
                                    color = "black",
                                    margin = margin(0, 15, 0, 0)),
        axis.line.x = element_line(size=1.2),
        axis.line = element_line(size=1.2))+
  xlab("Iteration")+ylab("Avg. accuracy")+
  scale_alpha(guide="none")+
  labs(color="")

##############################################################
#########################dimension reduction using shap values
#run an instance of xgboost to evaluate shap values
xgb.mod=xgboost(data = dtrain, 
                label = label, 
                max.depth=xgb.train$bestTune$max_depth, 
                eta=xgb.train$bestTune$eta, 
                nthread=4, 
                min_child_weight=xgb.train$bestTune$min_child_weight,
                scale_pos_weight=sumwneg/sumwpos, 
                eval_metric="auc", 
                eval_metric="rmse", 
                eval_metric="logloss",
                gamma=xgb.train$bestTune$gamma,
                nrounds=xgb.crv$best_iteration, 
                objective="binary:logistic",
                tree_method="hist",
                lambda=1,
                alpha=1)

#evaluate and plot feature importance
importance=xgb.importance(feature_names = colnames(dtrain), model = xgb.mod)
feat.label=importance$Feature[1:30]
feat.label=c("Closure = 1", "Work length", "Collision density", "Truck AADT",
             "ADT", "Closure length", "Closure coverage", "Work duration", "Peak AADT",
             "AADT", "Design speed", "Route ID = 10", "County = SJ", "Activity code = M90000",
             "Road width", "Work day = Wed.", "Surface type = C", "Work month = Sep.", "Work month = Jul.", 
              "Route ID = 210", "Work day = Fri.", "Work day = Thu.", 
             "Work day = Mon.", "Work day = Tue.", "Barrier type = E", "Work month = Jan.",
             "Work month = Dec.", "District = 8", "District = 4", "Work month = Aug.")

(gg=xgb.ggplot.importance(importance_matrix = importance[1:30,]))
gg+theme_ipsum(axis_title_just = "center")+
  theme(plot.title = element_blank(),
         axis.text.x = element_text(hjust = 0.5, 
                                    size=18, 
                                    family = "Century Gothic", 
                                    color = "black"),
         axis.title.x = element_text(size = 18, 
                                     family = "Century Gothic", 
                                     color = "black",
                                     margin = margin(15, 0, 0, 0)),
         axis.text.y = element_text(size=18, 
                                    family = "Century Gothic", 
                                    color = "black"),
         axis.title.y = element_text(size=18, 
                                     family = "Century Gothic", 
                                     color = "black",
                                     margin = margin(0, 15, 0, 0)),
        axis.line.x = element_line(size=1.2),
        legend.position = "none")+
  scale_x_discrete(labels=rev(feat.label))+
  xlab("Features")+
  ylab("Average relative contribution to minimization of the objective function")

#evaluate shap values
shap_values=shap.values(xgb_model = xgb.mod, X_train = dtest)
shap_values$mean_shap_score
#identify unimportant columns by shap value=0
#filter out the feature set by shap=0
rm_col=names(shap_values$mean_shap_score)[which(shap_values$mean_shap_score==0)]
train.df=as.data.frame(as.matrix(dtrain))
train.df=train.df[,!names(train.df)%in%rm_col]
train.ind=createDataPartition(train.df$collision_id, 
                              times = 1, 
                              p=0.5, 
                              list = FALSE)
train.df=train.df[train.ind,]
label=label[as.numeric(rownames(train.df))]

##############################################################
###################################feature selection by Boruta
xgb.brouta=Boruta(train.df,
                  y=as.numeric(as.factor(label))-1,
                  maxRuns=100, 
                  doTrace=2,
                  holdHistory=TRUE,
                  getImp=getImpXgboost,
                  max.depth=xgb.train$bestTune$max_depth, 
                  eta=xgb.train$bestTune$eta, 
                  nthread=4, 
                  min_child_weight=xgb.train$bestTune$min_child_weight,
                  scale_pos_weight=sumwneg/sumwpos, 
                  eval_metric="auc", 
                  eval_metric="rmse", 
                  eval_metric="logloss",
                  gamma=xgb.train$bestTune$gamma,
                  nrounds=xgb.crv$best_iteration, 
                  objective="binary:logistic",
                  tree_method="hist",
                  lambda=0,
                  alpha=0)

#extracting the result of Boruta algorithm
#keep confirmed and tentative features
boruta.df=attStats(xgb.brouta)
feature.imp=row.names(boruta.df)[which(boruta.df$decision!="Rejected")]
boruta.imp.df=as.data.frame(xgb.brouta$ImpHistory)
boruta.imp.df=boruta.imp.df[,names(boruta.imp.df)%in%feature.imp]
boruta.imp.df=melt(boruta.imp.df)
boruta.imp.df=cbind.data.frame(boruta.imp.df, 
                               decision=boruta.df$decision[match(boruta.imp.df$variable, 
                                                                 row.names(boruta.df))])

#rename the features
boruta.feat=c("Work duration", "Work length", "ADT", "Peak AADT",
              "AADT", "Truck AADT", "Collision density", "Lane closure = 1", 
              "Closure coverage", "Closure length")
fac=with(boruta.imp.df, reorder(variable, value, median, order = TRUE))
boruta.imp.df$variable=factor(boruta.imp.df$variable, levels = levels(fac))
ggplot(data = boruta.imp.df, aes(x=variable, y=value, fill=decision))+
  geom_boxplot()+
  coord_flip()+
  theme_ipsum(axis_title_just = "center")+
  theme(plot.title = element_blank(),
        axis.text.x = element_text(hjust = 0.5, 
                                   size=18, 
                                   family = "Century Gothic", 
                                   color = "black"),
        axis.title.x = element_text(size = 18, 
                                    family = "Century Gothic", 
                                    color = "black",
                                    margin = margin(15, 0, 0, 0)),
        axis.text.y = element_text(size=18, 
                                   family = "Century Gothic", 
                                   color = "black"),
        axis.title.y = element_text(size=18, 
                                    family = "Century Gothic", 
                                    color = "black",
                                    margin = margin(0, 15, 0, 0)),
        axis.line.x = element_line(size=1.2),
        axis.line.y = element_line(size=1.2),
        legend.title = element_text(size=18, 
                                    family = "Century Gothic", 
                                    color = "black"),
        legend.text = element_text(size=18, 
                                   family = "Century Gothic", 
                                   color = "black"),
        legend.position = "top")+
  scale_x_discrete(labels=boruta.feat)+
  xlab("Features")+
  ylab("Boruta Importance")

####################################################################
#shap analysis for Boruta features from a full model on the test set
#evaluate shap values
#
#attach("./bin/PMRF paper/xgb.boruta.RData", pos = 2)
#xgb.brouta=xgb.brouta
#detach("file:./bin/PMRF paper/xgb.boruta.RData")
#
shap_values=shap.values(xgb_model = xgb.mod, X_train = dtest)
shap_values$mean_shap_score
shap.feat=unique(boruta.imp.df$variable)
X=as.matrix(dtest)
X=as.data.frame(X)
X=X[, names(X) %in% shap.feat]

#removing some outliers for better visualization
inliers=which(X$work_duration<=0.2 & X$work_duration>=0 &
              X$work_length<=0.4 & X$work_length>=0 &
              X$road_adt<=0.8 & X$road_adt>=0 &
              X$peak_aadt<=0.3 & X$peak_aadt>=0 &
              X$aadt<=0.8 & X$aadt>=0 &
              X$truck_aadt<=0.4 & X$truck_aadt>=0 &
              X$collision_density11_12<=0.6 & X$collision_density11_12>=0 &
              X$closure_coverage<=1 & X$closure_coverage>=0 &
              X$closure_length<=0.2 & X$closure_length>=0)

#shap summary figure
shap_score_filtered=shap_values$shap_score
shap_score_filtered=setDF(shap_score_filtered)
shap_score_filtered=shap_score_filtered[inliers, names(shap_score_filtered) %in% shap.feat]

pp=shap.plot.summary.wrap2(shap_score = shap_score_filtered, 
                        X = X[inliers,],
                        dilute=20)
shap.names=c("Truck AADT", "Work duration", "Work length", "Closure length",
             "Peak AADT", "Collision density", "ADT", "Closure coverage",
             "AADT", "Lane closure = 1")
pp+theme_ipsum(axis_title_just = "center")+
  theme(plot.title = element_blank(),
        axis.text.x = element_text(hjust = 0.5, 
                                   size=18, 
                                   family = "Century Gothic", 
                                   color = "black"),
        axis.title.x = element_text(size = 18, 
                                    family = "Century Gothic", 
                                    color = "black",
                                    margin = margin(15, 0, 0, 0)),
        axis.text.y = element_text(size=18, 
                                   family = "Century Gothic", 
                                   color = "black"),
        axis.title.y = element_text(size=18, 
                                    family = "Century Gothic", 
                                    color = "black",
                                    margin = margin(0, 15, 0, 0)),
        axis.line.x = element_line(size=1.2),
        axis.line.y = element_line(size=1.2),
        legend.title = element_text(size=18, 
                                    family = "Century Gothic", 
                                    color = "black"),
        legend.text = element_text(size=18, 
                                   family = "Century Gothic", 
                                   color = "black"),
        legend.position = "right")+
  scale_x_discrete(labels=shap.names)+
  xlab("Features")+
  ylab("Shapley Values")+
  guides(colour=guide_colorbar(title = "Feature Value",
                               title.position = "top",
                               label.position = "left",
                               barwidth = 0.2,
                               barheight = 30))

####################################################################
#shap analysis for a model trained only by Boruta features

shap.df=train.df[, shap.feat]
shap.df=sparse.model.matrix(~.-1, shap.df)
xgb.mod=xgboost(data = shap.df, 
                label = label, 
                max.depth=xgb.train$bestTune$max_depth, 
                eta=xgb.train$bestTune$eta, 
                nthread=4, 
                min_child_weight=xgb.train$bestTune$min_child_weight,
                scale_pos_weight=sumwneg/sumwpos, 
                eval_metric="auc", 
                eval_metric="rmse", 
                eval_metric="logloss",
                gamma=xgb.train$bestTune$gamma,
                nrounds=xgb.crv$best_iteration, 
                objective="binary:logistic",
                max_delta_step=1,
                tree_method="hist",
                lambda=1,
                alpha=1)

#evaluate shap values
fin_shap_values=shap.values(xgb_model = xgb.mod, 
                            X_train = dtest)
fin_shap_values$mean_shap_score

#remove outliers for better visualization
inliers=which(X$work_duration<=6 & X$work_duration>=-1 &
                X$work_length<=6 & X$work_length>=0 &
                X$road_adt<=3 & X$road_adt>=0 &
                X$peak_aadt<=4 & X$peak_aadt>=0 &
                X$aadt<=3 & X$aadt>=0 &
                X$truck_aadt<=3 & X$truck_aadt>=0 &
                X$collision_density11_12<=3 & X$collision_density11_12>=0 &
                X$closure_coverage<=3 & X$closure_coverage>=-1 &
                X$closure_length<=5 & X$closure_length>=-1)

#shap summary plot
pp=shap.plot.summary.wrap2(shap_score = as.data.frame(as.matrix(fin_shap_values$shap_score))[inliers,], 
                           X = X[inliers,],
                           dilute=20)
pp$layers[[3]]=NULL
shap.names=c("Collision density", "ADT", "Work length", "Work duration", 
             "Peak AADT", "Truck AADT", "Closure length", "Lane closure",
             "AADT", "Closure coverage")
pp+theme_ipsum(axis_title_just = "center")+
  theme(plot.title = element_blank(),
        axis.text.x = element_text(hjust = 0.5, 
                                   size=18, 
                                   family = "Century Gothic", 
                                   color = "black"),
        axis.title.x = element_text(size = 18, 
                                    family = "Century Gothic", 
                                    color = "black",
                                    margin = margin(15, 0, 0, 0)),
        axis.text.y = element_text(size=18, 
                                   family = "Century Gothic", 
                                   color = "black"),
        axis.title.y = element_text(size=18, 
                                    family = "Century Gothic", 
                                    color = "black",
                                    margin = margin(0, 15, 0, 0)),
        axis.line.x = element_line(size=1.2),
        axis.line.y = element_line(size=1.2),
        legend.title = element_blank(),
        legend.text = element_text(size=18, 
                                   family = "Century Gothic", 
                                   color = "black"),
        legend.position = "right",
        legend.margin = margin(0, 0, 0, 0),
        legend.box.margin = margin(0, 0, 0, 0))+
  scale_x_discrete(labels=shap.names)+
  xlab("Features")+
  ylab("Shapley Values")+
  guides(colour=guide_colorbar(label.position = "right",
                               barwidth = 0.2,
                               barheight = 20))
####################################################################
################################################shap dependency plot
shap_long <- shap.prep(shap_contrib = fin_shap_values$shap_score, X_train = X)
shap_int <- shap.prep.interaction(xgb_mod = xgb.mod, X_train = X)

shap_long_inliers <- shap.prep(shap_contrib = as.data.frame(as.matrix(fin_shap_values$shap_score))[inliers,],
                               X_train = X[inliers, ])
shap_int_inliers <- shap.prep.interaction(xgb_mod = xgb.mod, X_train = X[inliers, ])

g1 <- shap.plot.dependence(data_long = shap_long_inliers, 
                           x = 'work_length', 
                           y = 'closure_id1', 
                           color_feature = 'closure_id1',
                           dilute=20) + 
  ggtitle("(A) SHAP values of Time trend vs. Time trend")

g2 <- shap.plot.dependence(data_long = shap_long_inliers,
                           data_int = shap_int_inliers,
                           x = 'work_length', 
                           y = 'closure_id1', 
                           color_feature = 'closure_id1',
                           dilute=20) + 
  ggtitle("(A) SHAP values of Time trend vs. Time trend")


#predict the test data
testing.df=fread("./bin/test(collision2class)_by.roadCondition.csv", sep = ",", header = TRUE)
y_test=testing.df$collision_id.1
rm(testing.df)
temp.predict=predict(xgb.mod, dtest)
temp.predict=as.numeric(temp.predict > 0.5)
confusionMatrix(as.factor(temp.predict), as.factor(testing.df$collision_id.1), positive = "1")

###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
#fit glm to the top features
featureSet=importance$Feature[1:54]
x=dtrain[, colnames(dtrain) %in% featureSet]
weights=ifelse(label==0, 1, sumwneg/sumwpos)

glm.mod=cv.glmnet(x=x, y=label, family="binomial", weights=weights, nfolds=5, 
                  type.logistic="modified.Newton", type.measure="auc", trace.it = 1)
#glm.mod=cv.glmnet(x=x, y=label, family="binomial", nfolds=5, 
#                  type.logistic="modified.Newton", type.measure="auc", trace.it = 1)
plot(glm.mod)
#coefficients(glm.mod, glm.mod$lambda.min)
coeff=coefficients(glm.mod, glm.mod$lambda.1se)
coeff=setDT(cbind.data.frame("coeff"=coeff[, 1], "feature"=row.names(coeff)), keep.rownames = TRUE)
fwrite(coeff, file = "coeff.csv", sep=",", append = FALSE, row.names = TRUE)
#temp.predict=predict(glm.mod, newx = dtest[, colnames(dtrain) %in% featureSet], s=glm.mod$lambda.min)
temp.predict=predict(glm.mod, newx = dtest[, colnames(dtrain) %in% featureSet], s=glm.mod$lambda.1se)
temp.predict=as.numeric(temp.predict > 0.5)
confusionMatrix(as.factor(temp.predict), as.factor(testing.df$collision_id), positive = "1")
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
#interpret each feature
dmtrain=xgb.DMatrix(data = dtrain, label=label)
dmtest=xgb.DMatrix(data=dtest, label=as.numeric(as.character(testing.df$collision_id)))

explainer = buildExplainer(xgb.mod, dmtrain, type="binary", base_score = 0.5, trees_idx = NULL)
pred.breakdown = explainPredictions(xgb.mod, explainer, dmtest)

cat('Breakdown Complete','\n')
weights = rowSums(pred.breakdown)
pred.xgb = 1/(1+exp(-weights))
cat(max(temp.predict-pred.xgb),'\n')

idx_to_get = as.integer(802)

test.df[idx_to_get, ]
showWaterfall(xgb.mod, explainer, dmtest, data.matrix(test.df) ,idx_to_get, type = "binary")
####### IMPACT AGAINST VARIABLE VALUE
plot(test.df[,closure_id], pred.breakdown[,closure_id1], cex=0.4, pch=16, 
     xlab = "Closure class", ylab = "closure class impact on log-odds")
plot(test.df[,work_length], pred.breakdown[,work_length], cex=0.4, pch=16, 
     xlab = "work length", ylab = "work length (miles) impact on log-odds")
plot(test.df[,collision_density11_12], pred.breakdown[,collision_density11_12], cex=0.4, pch=16, 
     xlab = "Collision density", ylab = "Collision density impact on log-odds")
plot(test.df[,truck_aadt], pred.breakdown[,truck_aadt], cex=0.4, pch=16, 
     xlab = "Collision density", ylab = "Collision density impact on log-odds")
plot(test.df[,road_adt], pred.breakdown[,road_adt], cex=0.4, pch=16, 
     xlab = "Road_adt", ylab = "Road adt impact on log-odds")
plot(test.df[,peak_aadt], pred.breakdown[,peak_adt], cex=0.4, pch=16, 
     xlab = "Road_adt", ylab = "Road adt impact on log-odds")

#cr <- colorRamp(c("blue", "red"))
#plot(test.df[,last_evaluation], pred.breakdown[,last_evaluation], col = rgb(cr(round(test.df[,satisfaction_level])), max=255), cex=0.4, pch=16, xlab = "Last evaluation", ylab = "Last evaluation impact on log-odds")