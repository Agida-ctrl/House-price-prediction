## Kaggle house price prediction using Xgboost model

#libraries
library(tidymodels)
library(xgboost)
library(skimr)

train<-read.csv("House_train.csv")
test<-read.csv("House_test.csv")
skim(train)
skim(test)

train$MSSubClass<-as.factor(train$MSSubClass)
train$Alley[is.na(train$Alley)]<-"No Alley"
train$OverallQual<-as.factor(train$OverallQual)
train$OverallCond<-as.factor(train$OverallCond)
train$MasVnrType[is.na(train$MasVnrType)]<-"CBlock"
train$MasVnrArea[is.na(train$MasVnrArea)]<-mean(train$MasVnrArea,
                                                na.rm = T)
train$BsmtQual[is.na(train$BsmtQual)]<-"No Basement"
train$BsmtCond[is.na(train$BsmtCond)]<-"No Basement"
train$BsmtExposure[is.na(train$BsmtExposure)]<-"No Basement"
train$BsmtFinType1[is.na(train$BsmtFinType1)]<-"No Basement"
train$BsmtFinType2[is.na(train$BsmtFinType2)]<-"No Basement"
train$Fireplaces<-as.factor(train$Fireplaces)
train$FireplaceQu[is.na(train$FireplaceQu)]<-"No fireplace"
train$GarageType[is.na(train$GarageType)]<-"No Garage"
train$GarageFinish[is.na(train$GarageFinish)]<-"No Garage"
train$GarageQual[is.na(train$GarageQual)]<-"No Garage"
train$GarageCond[is.na(train$GarageCond)]<-"No Garage"
train$PoolQC[is.na(train$PoolQC)]<-"No pool"
train$Fence[is.na(train$Fence)]<-"No fence"
train$MiscFeature[is.na(train$MiscFeature)]<-"None"
train$MoSold<-as.factor(train$MoSold)
train$YrSold<-as.factor(train$YrSold)
train$GarageYrBlt[is.na(train$GarageYrBlt)]<-0

train<-train %>% mutate_if(is.character,factor) %>% 
        na.omit()

test$MSSubClass<-as.factor(test$MSSubClass)
test$Alley[is.na(test$Alley)]<-"No Alley"
test$OverallQual<-as.factor(test$OverallQual)
test$OverallCond<-as.factor(test$OverallCond)
test$MasVnrType[is.na(test$MasVnrType)]<-"CBlock"
test$MasVnrArea[is.na(test$MasVnrArea)]<-mean(test$MasVnrArea,
                                              na.rm = T)
test$BsmtQual[is.na(test$BsmtQual)]<-"No Basement"
test$BsmtCond[is.na(test$BsmtCond)]<-"No Basement"
test$BsmtExposure[is.na(test$BsmtExposure)]<-"No Basement"
test$BsmtFinType1[is.na(test$BsmtFinType1)]<-"No Basement"
test$BsmtFinType2[is.na(test$BsmtFinType2)]<-"No Basement"
test$Fireplaces<-as.factor(test$Fireplaces)
test$FireplaceQu[is.na(test$FireplaceQu)]<-"No fireplace"
test$GarageType[is.na(test$GarageType)]<-"No Garage"
test$GarageFinish[is.na(test$GarageFinish)]<-"No Garage"
test$GarageQual[is.na(test$GarageQual)]<-"No Garage"
test$GarageCond[is.na(test$GarageCond)]<-"No Garage"
test$PoolQC[is.na(test$PoolQC)]<-"No pool"
test$Fence[is.na(test$Fence)]<-"No fence"
test$MiscFeature[is.na(test$MiscFeature)]<-"None"
test$MoSold<-as.factor(test$MoSold)
test$YrSold<-as.factor(test$YrSold)
test$GarageYrBlt[is.na(test$GarageYrBlt)]<-0

test<-test %>% mutate_if(is.character,factor)

skim(test_house)
skim(train_house)

splits<-initial_split(train,strata = SalePrice)
h_train<-training(splits)
h_test<-testing(splits)




#without preprocessing

xgb_spec<-boost_tree(mode = "regression",
                     engine = "xgboost",
                     trees = 1000,tree_depth = tune(),
                     min_n = tune(),
                     loss_reduction = tune(),
                     sample_size = tune(),
                     mtry = tune(),
                     learn_rate = tune())


xgb_grid<-grid_latin_hypercube(
        tree_depth(),
        min_n(),
        loss_reduction(),
        sample_size = sample_prop(),
        finalize(mtry(),h_train),
        learn_rate(),
        size = 20
)

xgb_wf<- workflow() %>% 
        add_formula(SalePrice ~ .) %>%
        add_model(xgb_spec)
xgb_wf

cv_folds<-vfold_cv(h_train,strata = SalePrice)
cv_folds

doParallel::registerDoParallel()
xgb_res<-tune_grid(
        xgb_wf,
        resamples = cv_folds,
        grid = xgb_grid,
        control = control_grid(save_pred = TRUE)
)

xgb_res %>% collect_metrics() %>% 
        filter(.metric=="rmse") %>%
        select(mean,mtry:sample_size) %>%
        pivot_longer(mtry:sample_size,
                     names_to = "para", values_to = "value") %>%
        ggplot(aes(value,mean,color = para)) + 
        geom_point(show.legend = F)+
        facet_wrap(~para,scales = "free_x")


best_rmse<-select_best(xgb_res,"rmse")
final_xgb<-finalize_workflow(xgb_wf,best_rmse)
library(vip)

final_xgb %>% fit(data = h_train) %>%
        extract_fit_parsnip() %>% vip(geom = "point")


final_res<-last_fit(final_xgb,splits)
final_res %>% collect_metrics()

pred<-final_res %>% collect_predictions()


#Observed values versus predicted values
# It is a good idea to plot the values on a common scale.
axisRange <- extendrange(c(pred$SalePrice, pred$.pred))
plot(pred$SalePrice, pred$.pred,
        ylim = axisRange,
        xlim = axisRange)
# Add a 45 degree reference line
abline(0, 1, col = "darkgrey", lty = 2)

# Predicted values versus residuals
resid<-pred$SalePrice-pred$.pred
plot(pred$.pred, resid, ylab = "residual")
abline(h = 0, col = "darkgrey", lty = 2)


test_pred<-final_xgb %>% fit(data = h_train) %>% predict(new_data = test)

submission<-data.frame(id=test$Id,saleprice=test_pred$.pred)
write.csv(submission,"submission.csv",row.names = F)
