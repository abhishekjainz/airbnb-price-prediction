
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

# Quick look on the layers
tmap_mode("view")
qtm(newdf)
qtm(newmalls)
qtm(CA)
qtm(county)
qtm(places)


### Convert distance in Degrees to Meters
temp <- st_as_sf(newdf)
st_crs(temp) <- 4326
temp <- st_transform(temp, crs = 7801)
buff <- st_buffer(temp, dist = 1000)
buff <- st_transform(buff, crs = 4326)
union <- st_union(buff) %>% st_make_valid()


### Combine all layers in one view 
### Note: Might experience lag when navigating the plot
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
finaldf <- newdf %>% st_transform(crs = 4326)
finalmalls <- newmalls %>% st_transform(crs = 4326)
intervals <- seq(500, 2000, 500)

### Method 1: within distance split into columns
finaldf <- finaldf %>% 
  mutate(within_500 = lengths(st_is_within_distance(x = .,
                                                     y = finalmalls,
                                                     dist = 500)))
finaldf <- finaldf %>% 
  mutate(within_1000 = lengths(st_is_within_distance(x = .,
                                                     y = finalmalls,
                                                     dist = 1000)))
finaldf <- finaldf %>% 
  mutate(within_1500 = lengths(st_is_within_distance(x = .,
                                                     y = finalmalls,
                                                     dist = 1500)))
finaldf <- finaldf %>% 
  mutate(within_2000 = lengths(st_is_within_distance(x = .,
                                                    y = finalmalls,
                                                    dist = 2000)))

finaldf_without_geo <- finaldf %>% st_drop_geometry()

# Test
testrow = 14
test <- finaldf[testrow,]

temp_single <- st_as_sf(test)
st_crs(temp_single) <- 4326
point_temp <- st_transform(temp_single, crs = 7801)
buff_single <- st_buffer(temp_single, dist = 2000) %>% st_transform(crs = 4326)

tmap_mode("view")
tm_shape(CA) +
  tm_borders("black") +
  tm_shape(buff_single) + tm_polygons("yellow", alpha = 0.6) +
  tm_shape(newmalls) + tm_dots("blue", size = 0.01,
                               popup.vars = c('Name: ' = 'title', 
                                              'Description' = 'attributes', 
                                              'Rating: ' = 'rating',
                                              'Review Count: ' = 'reviewCount',
                                              'Address: ' = 'address',
                                              'City: ' = 'city',
                                              'Website: ' = 'website')) + 
  tm_shape(test) + tm_dots("red", size = 0.01) 


sprintf("Point %.0f has %.0f malls within 2km buffer", 14, finaldf_without_geo[testrow,40]) 


### Method 2: Find all intervals at one go, consolidated in one column
faster_df <- map_dfr(intervals, 
                     ~ testdf %>% 
                       mutate(within = .x, 
                              n = lengths(st_is_within_distance(x = .,
                                                    y = testmalls,
                                                    dist = .x))) %>%
                       st_drop_geometry())

st_crs(newdf)


##########################################################
##### LOCAL ATTRACTTIONS WITHIN BUFFER FROM LISTINGS #####
##########################################################





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
