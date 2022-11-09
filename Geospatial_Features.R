
library(sf)
library(ggmap)
library(maptools)
library(tmap)
library(dplyr)
library(tidyverse)
library(mapview)
library(future)
library(rvest)
library(raster)
library(rgeos)
library(GISTools)
library(leaflet)
library(rgdal)
library(rmapshaper)
library(ggpubr)

path = "data/"

#####################
##### READ DATA #####
#####################
CA <- read_sf(dsn = paste(path, "geospatial/ca-state-boundary/", sep=""), 
              layer = "CA_State_TIGER2016")
county <- read_sf(dsn = paste(path, "geospatial/CA_Counties/", sep=""), 
                  layer = "CA_Counties_TIGER2016")
places <- read_sf(dsn = paste(path, "geospatial/ca-places-boundaries/", sep=""), 
                  layer = "CA_Places_TIGER2016")

df <- read.csv(file = paste(path, "cleaned_data.csv", sep=""))
malls <- read.csv(file = paste(paste(path, "geospatial/", sep=""), 
                               "malls.csv", sep=""))
attractions <- read.csv(file = paste(paste(path, "geospatial/", sep=""), 
                                     "local_attractions.csv", sep=""))

projcrs <- "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"
newdf <- st_as_sf(x=df, coords = c("longitude", "latitude"), crs = projcrs)
newmalls <- st_as_sf(x=malls, 
                     coords = c("longitude", "latitude"), 
                     crs = projcrs)
newattractions <- st_as_sf(x=attractions, 
                           coords = c("longitude", "latitude"), 
                           crs = projcrs)

### Quick look on the layers
tmap_mode("view")
qtm(newdf)
qtm(newmalls)
qtm(CA)
qtm(county)
qtm(places)

########################
##### BUFFER ZONES #####
########################
### Convert distance in Degrees to Meters
temp <- st_as_sf(newdf)
st_crs(temp) <- 4326
temp <- st_transform(temp, crs = 7801)
buff <- st_buffer(temp, dist = 1000)
buff <- st_transform(buff, crs = 4326)
union <- st_union(buff) %>% st_make_valid()


### Combine all layers in one view 
### Note: Might take some time and experience lag when navigating the plot
tmap_mode("plot")
tmap_options(check.and.fix = TRUE)
tm_shape(CA) +
  tm_borders("black") + tm_fill('white') +
  tm_shape(union) + tm_polygons("yellow", alpha = 1.0) +
  tm_shape(newdf) + tm_dots("black", size = 0.01) +
  tm_shape(newmalls) + tm_dots("blue", size = 0.01)


#############################################
##### MALLS WITHIN BUFFER FROM LISTINGS #####
#############################################
### Convert df to crs = 4326
finaldf <- newdf %>% st_transform(crs = 4326)

### Convert data to crs = 7801 for accurate distance measurement
finalmalls <- newmalls %>% st_transform(crs = 7801)

### Method 1: within distance split into columns
finaldf <- finaldf %>% 
  mutate(num_mall_within_500 = lengths(st_is_within_distance(x = .,
                                                     y = finalmalls,
                                                     dist = 500)))
finaldf <- finaldf %>% 
  mutate(num_mall_within_1000 = lengths(st_is_within_distance(x = .,
                                                     y = finalmalls,
                                                     dist = 1000)))
finaldf <- finaldf %>% 
  mutate(num_mall_within_1500 = lengths(st_is_within_distance(x = .,
                                                     y = finalmalls,
                                                     dist = 1500)))
finaldf <- finaldf %>% 
  mutate(num_mall_within_2000 = lengths(st_is_within_distance(x = .,
                                                    y = finalmalls,
                                                    dist = 2000)))


##########################################################
##### LOCAL ATTRACTTIONS WITHIN BUFFER FROM LISTINGS #####
##########################################################
### Convert data to crs = 7801 for accurate distance measurement
finalattractions <- newattractions %>% st_transform(crs = 7801)

### Method 1: within distance split into columns
finaldf <- finaldf %>% 
  mutate(num_attraction_within_500 = lengths(st_is_within_distance(x = .,
                                                             y = finalattractions,
                                                             dist = 500)))
finaldf <- finaldf %>% 
  mutate(num_attraction_within_1000 = lengths(st_is_within_distance(x = .,
                                                              y = finalattractions,
                                                              dist = 1000)))
finaldf <- finaldf %>% 
  mutate(num_attraction_within_1500 = lengths(st_is_within_distance(x = .,
                                                              y = finalattractions,
                                                              dist = 1500)))
finaldf <- finaldf %>% 
  mutate(num_attraction_within_2000 = lengths(st_is_within_distance(x = .,
                                                              y = finalattractions,
                                                              dist = 2000)))

###############################
##### ADD LOGICAL COLUMNS #####
###############################
finaldf$mall_within_500 <- ifelse(finaldf$num_mall_within_500 > 0, 1, 0)
finaldf$mall_within_1000 <- ifelse(finaldf$num_mall_within_1000 > 0, 1, 0)
finaldf$mall_within_1500 <- ifelse(finaldf$num_mall_within_1500 > 0, 1, 0)
finaldf$mall_within_2000 <- ifelse(finaldf$num_mall_within_2000 > 0, 1, 0)
finaldf$attraction_within_500 <- ifelse(finaldf$num_attraction_within_500 > 0, 1, 0)
finaldf$attraction_within_1000 <- ifelse(finaldf$num_attraction_within_1000 > 0, 1, 0)
finaldf$attraction_within_1500 <- ifelse(finaldf$num_attraction_within_1500 > 0, 1, 0)
finaldf$attraction_within_2000 <- ifelse(finaldf$num_attraction_within_2000 > 0, 1, 0)


###################
##### TESTING #####
###################
# Visualise the listings and facilities on the map and verify with the data 
# to see if it aligns

testrow = 7649
test <- finaldf %>% as.data.frame() %>%
  subset(X == testrow)

temp_single <- st_as_sf(test)
st_crs(temp_single) <- 4326
temp_single <- st_transform(temp_single, crs = 7801)
buff_single <- st_buffer(temp_single, dist = 1000) %>% st_transform(crs = 4326)

tmap_mode("view")
tm_shape(CA) +
  tm_borders("black") +
  tm_shape(buff_single) + tm_polygons("yellow", alpha = 0.6) +
  tm_shape(finalmalls) + tm_dots("blue", size = 0.01,
                               popup.vars = c('Name: ' = 'title', 
                                              'Description' = 'attributes', 
                                              'Rating: ' = 'rating',
                                              'Review Count: ' = 'reviewCount',
                                              'Address: ' = 'address',
                                              'City: ' = 'city',
                                              'Website: ' = 'website')) + 
  tm_shape(finalattractions) + tm_dots("purple", size = 0.01,
                               popup.vars = c('Name: ' = 'title', 
                                              'Description' = 'attributes', 
                                              'Rating: ' = 'rating',
                                              'Review Count: ' = 'reviewCount',
                                              'Address: ' = 'address',
                                              'City: ' = 'city',
                                              'Website: ' = 'website')) +
  tm_shape(st_as_sf(test)) + tm_dots("red", size = 0.01) 


sprintf("Listing %.0f has %.0f malls and %.0f attractions within 1km buffer", 
        test$X, test$num_mall_within_1000, test$num_attraction_within_1000)

# Output: Listing 7649 has 4 malls and 4 attractions within 1km buffer

### Split geometry column to Lat/Long value and export results to csv
export <- finaldf %>%
  mutate(longitude = unlist(map(finaldf$geometry,1)),
         latitude = unlist(map(finaldf$geometry,2))) %>%
  st_drop_geometry()

write.csv(export, paste(path,"cleaned_data_with_geo.csv",sep = ""))

#######################
##### EXTRA CODES #####
#######################
### Method 2: Find all intervals at one go, consolidated in one column
# intervals <- seq(500, 2000, 500)
# faster_df <- map_dfr(intervals, 
#                      ~ testdf %>% 
#                        mutate(within = .x, 
#                               n = lengths(st_is_within_distance(x = .,
#                                                                 y = testmalls,
#                                                                 dist = .x))) %>%
#                        st_drop_geometry())
# 
# st_crs(newdf)


### Naive Method by buffering all points one by one, extremely inefficient
# finaldf <- newdf %>% as.data.frame()
# finaldf$within_500 = 0
# finaldf$within_1000 = 0
# finaldf$within_1500 = 0
# finaldf$within_2000 = 0
# 
# numRow <- nrow(finaldf)
# for (i in 1:numRow) {
#   point <- finaldf[i,37]
#   temp_single <- st_as_sf(test)
#   st_crs(temp_single) <- 4326
#   point_temp <- st_transform(temp_single, crs = 7801)
#   
#   buff_single_500 <- st_buffer(temp_single, dist = 500) %>% 
#     st_transform(crs = 4326)
#   buff_single_1000 <- st_buffer(temp_single, dist = 1000) %>% 
#     st_transform(crs = 4326)
#   buff_single_1500 <- st_buffer(temp_single, dist = 1500) %>% 
#     st_transform(crs = 4326)
#   buff_single_2000 <- st_buffer(temp_single, dist = 2000) %>% 
#     st_transform(crs = 4326)
#   
#   over_single_2000 <- st_intersection(buff_single_2000, newmalls)
#   
#   if (!is.null(nrow(over_single_2000))) {
#     finaldf$within_2000[i] = nrow(over_single_2000)
#     over_single_1500 <- st_intersection(buff_single_1500, newmalls)
#     if (!is.null(nrow(over_single_1500))) {
#       finaldf$within_1500[i] = nrow(over_single_1500)
#       over_single_1000 <- st_intersection(buff_single_1000, newmalls)
#       if (!is.null(nrow(over_single_1000))) {
#         finaldf$within_1000[i] = nrow(over_single_1000)
#         over_single_500 <- st_intersection(buff_single_500, newmalls)
#         if (!is.null(nrow(over_single_500))) {
#           finaldf$within_500[i] = nrow(over_single_500)
#         }
#       }
#     }
#   }
#   sprintf("Point %.0f completed. %.0f points to go.", i, numRow - i) 
# }
# print("All points completed!")
