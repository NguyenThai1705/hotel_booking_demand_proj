## Data Science Hotel Booking Cancellation Predict: Project Overview
An Online travel booking company is suffering from loss in revenue because of the uncertain booking cancellation of its customers. The company want to know which customer will cancel the booking.  
As a Data-scientist we have to help the company to predict whether the customer will cancel the booking or not.  
We have to do some data analysis to answer some questions and we have to work on Machine Learning models to help predict whether the customer will cancel the booking or not. We will focus on Exploratory Data Analysis for answering business questions first and then move on to the prediction approach.  
In this project I:
* Analyzed the hotel booking dataset to discover answers to business questions in hotel market.
* Created a tool that predicts the possibility of canceling a hotel reservation by a customer.
* Build a client facing API using Flask.

## Code and Resources Used
**Python Version:** 3.8  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, plotly, flask, json, pickle.  
**For Web Framework Requirements:** '''pip install -r requirements.txt'''  
**Hotel booking dataset:** https://www.kaggle.com/jessemostipak/hotel-booking-demand  

## Hotel booking Dataset 
119k records of customers who sent requests to the hotel booking servers. With each records, we got the following:
* Hotel (Resort Hotel and City Hotel)
* is_canceled - Value indicating if the booking was canceled (1) or not (0)
* lead_time - Number of days that elapsed between the entering date of the booking into the PMS and the arrival date
* arrival_date_year
* arrival_date_month
* arrival_date_week_number - Week number of year for arrival date
* arrival_date_day_of_month
* stays_in_weekend_nights - Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel
* stays_in_week_nights - Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel
* adults 
* children
* babies
* meal
* country
* market_segment
* distribution_channel
* is_repeated_guest
* previous_cancellations
* previous_bookings_not_canceled
* reserved_room_type
* assigned_room_type
* booking_changes
* deposit_type
* agent
* company
* days_in_waiting_list
* customer_type
* adr - Average Daily Rate as defined by dividing the sum of all lodging transactions by the total number of staying nights
* required_car_parking_spaces
* total_of_special_requests
* reservation_status
* reservation_status_date

## Data Cleaning
After take off the dataset, i needed to clean it up so that it was unable for EDA and our model. I made the following changes:
* Changed Null values in agent and company to 'Not Applicable' (Because in this dataset, "Null" is presented as one of the categories).
* Made a interpretation for meal.

## EDA 
I look at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights from the visualization.

![alt text](https://github.com/NguyenThai1705/hotel_booking_demand_proj/blob/main/guests_by_country.png "Top 20 countries with the most guests by the resort hotel")
![alt text](https://github.com/NguyenThai1705/hotel_booking_demand_proj/blob/main/guests_by_month.png "Number of guests in each month")
![alt text](https://github.com/NguyenThai1705/hotel_booking_demand_proj/blob/main/meal_by_hotel.png "types of meal by the resort hotel")
## Model building

First, I transformed the categorical variables into dummies variables. I also splited data into train and test datasets with a test size of 20%.
I used **Multiple Logistic Regression** to classify the cancellation (with 2 variables 0 vs 1)

## Model performance
100% Test set accuracy

## Productionization
I built a flask API endpoint that was hosted a local webserver. The API endpoint takes in a request with a list of values from a job listing and returns an estimated salary.



