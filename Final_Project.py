import inline as inline
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import lightgbm as lgb
import numpy as np
import spark as spark
from sklearn import cluster
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale


import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class Modeler:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def split_data(self, test_size=0.3):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=test_size)

    def plotModel(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Predictions vs. True Values", fontsize=14, y=1)
        plt.subplots_adjust(top=0.93, wspace=0)
        ax1.scatter(self.y_test, self.test_preds, s=2, alpha=0.5)
        ax1.plot(list(range(0, int(self.y_test.max()) + 10)), color='black', linestyle='--')
        ax1.set_title("Test Set")
        ax1.set_xlabel("True Values")
        ax1.set_ylabel("Predictions")
        ax2.scatter(self.y_train, self.train_preds, s=2, alpha=0.5)
        ax2.plot(list(range(0, int(self.y_train.max()) + 10)), color='black', linestyle='--')
        ax2.set_title("Train Set")
        ax2.set_xlabel("True Values")
        ax2.set_ylabel("")
        plt.show()

    def linreg(self):
        self.model = LinearRegression(normalize=True)
        self.model.fit(self.x_train, self.y_train)
        self.train_preds = self.model.predict(self.x_train)
        self.test_preds = self.model.predict(self.x_test)

        train_mse = mse(self.y_train, self.train_preds)
        test_mse = mse(self.y_test, self.test_preds)
        print(train_mse, test_mse)
        print("Train R2", r2_score(self.y_train, self.train_preds))
        print("Test R2", r2_score(self.y_test, self.test_preds))

    def lassoer(self):
        self.model = Lasso(normalize=True)
        self.model.fit(self.x_train, self.y_train)
        self.train_preds = self.model.predict(self.x_train)
        self.test_preds = self.model.predict(self.x_test)

        train_mse = mse(self.y_train, self.train_preds)
        test_mse = mse(self.y_test, self.test_preds)
        print(train_mse, test_mse)
        print("Train R2", r2_score(self.y_train, self.train_preds))
        print("Test R2", r2_score(self.y_test, self.test_preds))

    def ridger(self):
        self.model = Ridge(normalize=True)
        self.model.fit(self.x_train, self.y_train)
        self.train_preds = self.model.predict(self.x_train)
        self.test_preds = self.model.predict(self.x_test)

        train_mse = mse(self.y_train, self.train_preds)
        test_mse = mse(self.y_test, self.test_preds)
        print(train_mse, test_mse)
        print("Train R2", r2_score(self.y_train, self.train_preds))
        print("Train R2", r2_score(self.y_test, self.test_preds))

    def spliner(self):
        pass


    def polynomial(self):
        self.model = np.polyadd(self.x_train['beds'], self.y_train)





    def principalComponentAnalysis(self):
        #pca = PCA(svd_solver = 'full')
        #pca.fit(self.x_train)
        #print(pca.explained_variance_ratio_)
        pca = PCA(n_components=3) # project from 171 to 2 dimensions
        #pca.fit(self.x_train)
        projected = pca.fit_transform(self.x)

        #X_pca = pca.transform(self.x_train)
        print("original shape:   ", self.x.shape)
        print("transformed shape:", projected.shape)
        #ax = plt.axes(projected='3d')
        #Axes3D.scatter(projected[:, 0], projected[:, 1], projected[:, 2], c = 'red')
        #ax.plot3D(projected[:, 0], projected[:, 1], projected[:, 2],edgecolor='none', alpha=0.5,
        #            cmap=plt.cm.get_cmap('Accent', 117))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax.plot(projected[:, 0], projected[:, 1], projected[:, 2])
        ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2], c='r', marker='o')

        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        #plt.colorbar()
        plt.show()

        percent_variance = np.round(pca.explained_variance_ratio_ * 100, decimals=2)
        columns = ['PC1', 'PC2', 'PC3']
        plt.bar(x=range(1, 4), height=percent_variance, tick_label=columns)
        plt.ylabel('Percentate of Variance Explained')
        plt.xlabel('Principal Component')
        plt.title('PCA Scree Plot')
        plt.show()

        #print(self.x_train)
        #print(self.y_train)

        plt.scatter(projected[:, 0],self.y)
        plt.xlabel('Component 1')
        plt.ylabel('Price')
        plt.show()

        # plt.scatter(projected[:, 0], projected[:, 1],
        #             edgecolor='none', alpha=0.5,
        #             cmap=plt.cm.get_cmap('Accent', 117))
        # plt.xlabel('component 1')
        # plt.ylabel('component 2')
        # plt.colorbar()
        # plt.show()




        # print("this is a test!")
        # X_new = pca.inverse_transform(X_pca)
        # plt.scatter(X_new[:, 0], X_new[:, 1], X_new[:, 2])
        # plt.axis('equal')
        # plt.show()

        #print(pca.singular_values_)










        # Visualizing the Polymonial Regression results
    def viz_polymonial(self):
        print("this is being read by the computer!!!")
        poly_reg = PolynomialFeatures(degree=10)
        X_poly = poly_reg.fit_transform(self.x_train)
        pol_reg = LinearRegression()
        pol_reg.fit(X_poly, self.y_train)
        print("this is being read as well!!")
        # plt.scatter(self.x_train, self.y_train, color='red')
        # plt.plot(self.x_train, pol_reg.predict(poly_reg.fit_transform(self.x_train)), color='blue')
        # plt.title('Truth or Bluff (Linear Regression)')
        # plt.xlabel('Position level')
        # plt.ylabel('Salary')
        # plt.show()

        # plt.scatter(self.x_train, self.y_train, color='blue')
        #
        # plt.plot(self.x_train, lin2.predict(poly.fit_transform(self.x_train)), color='red')
        # plt.title('Polynomial Regression')
        # plt.xlabel('Temperature')
        # plt.ylabel('Pressure')
        #
        # plt.show()


    # def cross_validation(self, cv=10):
    #     kf = KFold(n_splits=cv, random_state=None, shuffle=True)
    #     rmse_scores = []
    #     alphas = list(np.logspace(-15, 15, 151, base=2))
    #     for train_index, test_index in kf.split(self.x_train):
    #         X_trn, X_tst = self.x_train.iloc[train_index], self.x_train.iloc[test_index]
    #         y_trn, y_tst = self.y_train.iloc[train_index], self.y_train.iloc[test_index]
    #
    #         ### straight ridge
    #         ridge1 = RidgeCV(alphas=alphas, cv=10, normalize=True)
    #         ridge1.fit(self.x_train, self.y_train)
    #
    #         y_fit = ridge1.predict(X_trn)
    #         resid = y_trn - y_fit
    #
    #         model = lgb.LGBMRegressor(objective='regression')
    #
    #         tuning_parameters = {
    #             'learning_rate': [0.01, 0.05, 0.1],
    #             'n_estimators': [250, 500, 750, 1000, 1500],
    #             'max_depth': [2, 3, 4],
    #             'subsample': [0.6, 0.8, 1.0],
    #         }
    #
    #         gb_search = RandomizedSearchCV(model, tuning_parameters, n_iter=16, cv=5,
    #                                        return_train_score=False, n_jobs=4, random_state=42)
    #
    #         gb_search.fit(X_trn, resid)
    #
    #         abst = gb_search.best_estimator_
    #
    #         y_pred = ridge1.predict(X_tst) + abst.predict(X_tst)
    #         mse = np.sum((y_pred - y_tst) ** 2) / len(y_pred)
    #         rmse_score = np.sqrt(mse)
    #         print(rmse_score)
    #         rmse_scores.append(rmse_score)
    #
    #     print("testing this function")

    # def CubicSpline(self):
    #     self.model = csaps.MultivariateCubicSmoothingSpline(self.x_train, self.y_train, smooth=0.988)
    #     self.model.fit(self.x_train, self.y_train)
        # self.train_preds = self.model.predict(self.x_train)
        # self.test_preds = self.model.predict(self.x_test)
        # train_mse = mse(self.y_train, self.train_preds)
        # test_mse = mse(self.y_test, self.test_preds)
        # print(train_mse, test_mse)
        # print("Train R2", r2_score(self.y_train, self.train_preds))
        # print("Train R2", r2_score(self.y_test, self.test_preds))

class Listings:
    def __init__(self, filename='/Users/joelnorthrup/Desktop/listings_summary.csv', groupAmenities=False, DATT=True):
        # pd.set_option('display.max_columns', 500)
        self.readListings(filename)
        self.groupAmenities = groupAmenities
        if DATT:
            self.dropCols()
            self.cleaner()

    def corrHeatMap(self):
        corr = self.df.select_dtypes(include='number').drop('id', axis=1).corr()
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.tril_indices_from(mask)] = True
        f, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3, center=0, square=True, linewidths=0.5,
                    cbar_kws={'shrink': 0.5})
        plt.show()

    def readListings(self, filename):

        # suppress low_memory warning--file isn't large enough for
        # any significant processing impact.
        self.df = pd.read_csv(filename, low_memory=False)
        self.df.set_index('id')
        self.columns = self.df.columns.values.tolist()

    def longLat(self, k=10, showplot=False):
        '''Use KMeans to categorize lat/long into our own version of neighborhoods.
        Set k = -1 to graph error for k=1:20. Otherwise, set K and returns kmeans.
        8 was found to be a good number.'''
        if k <= 0:
            print("Fitting 20 ks")
            cluster_sum_squares = []
            for i in range(1, 20):
                print("iteration", i)
                kmeans = KMeans(n_clusters=i, init='k-means++')
                kmeans.fit(self.df[['latitude', 'longitude']].copy())
                cluster_sum_squares.append(kmeans.inertia_)
            plt.plot(range(1, 20), cluster_sum_squares)
            plt.xlabel("# Clusters")
            plt.ylabel("Cluster Sum of Squares")
            plt.show()
            return kmeans

        kmeans = KMeans(n_clusters=k, init='k-means++')
        labels = kmeans.fit_predict(self.df[['latitude', 'longitude']].copy())
        if showplot:
            sns.scatterplot(self.df.latitude, self.df.longitude, alpha=0.3)
            sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1])
            plt.show()
        self.df['kmeans_neighborhoods'] = labels
        self.df.kmeans_neighborhoods = self.df.kmeans_neighborhoods.astype('category')
        self.df.drop(['latitude', 'longitude'], axis=1)
        return kmeans

    def nullVals(self, column):
        return self.df[column].isna().sum()

    def fillNulls(self, column, newNull):
        self.df[column] = self.df[column].fillna(newNull)

    def dollarToFloat(self, column):
        if type(self.df[column][0]) != str:
            print("Data is not a string")
        else:
            self.df[column] = self.df[column].str.replace('$', '').str.replace(',', '').astype(float)

    def dropCols(self, toDrop=[
        'summary', 'space', 'description', 'neighborhood_overview',
        'notes', 'transit', 'access', 'interaction',
        'house_rules', 'thumbnail_url', 'host_about', 'license',
        'listing_url', 'scrape_id', 'last_scraped', 'name',
        'medium_url', 'picture_url', 'xl_picture_url', 'host_id', 'host_since',
        'host_location', 'host_acceptance_rate', 'host_neighbourhood',
        'host_listings_count', 'host_total_listings_count',
        'host_url', 'host_name', 'host_thumbnail_url', 'host_picture_url',
        'neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed',
        'city', 'state', 'street', 'zipcode', 'market', 'smart_location', 'country_code',
        'country', 'is_location_exact',
        'weekly_price', 'monthly_price', 'square_feet',
        'minimum_nights',
        'maximum_nights',
        'calendar_updated', 'calendar_last_scraped', 'first_review', 'last_review',
        'jurisdiction_names', 'experiences_offered',
        'host_verifications']):
        self.df = self.df.drop(toDrop, axis=1)
        self.columns = self.df.columns.values.tolist()

#    'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights','number_of_reviews_ltm',
    def cleaner(self):

        # convert property types to an index
        self.property_types = self.df.property_type.unique().tolist()
        self.property_types_index = {}
        for i in range(len(self.property_types)):
            self.property_types_index[self.property_types[i]] = i
        self.df.property_type = self.df.property_type.apply(lambda x: self.property_types_index[x]).astype('category')

        # convert room types to an index
        self.room_types = self.df.room_type.unique().tolist()
        self.room_types_index = {}
        for i in range(len(self.room_types)):
            self.room_types_index[self.room_types[i]] = i
        self.df.room_type = self.df.room_type.apply(lambda x: self.room_types_index[x]).astype('category')

        # convert bed types to an index
        self.bed_types = self.df.bed_type.unique().tolist()
        self.bed_types_index = {}
        for i in range(len(self.bed_types)):
            self.bed_types_index[self.bed_types[i]] = i
        self.df.bed_type = self.df.bed_type.apply(lambda x: self.bed_types_index[x]).astype('category')

        # convert cancellation policy to an index
        self.cancellation_policies = self.df.cancellation_policy.unique().tolist()
        self.cancellation_policies_index = {}
        for i in range(len(self.cancellation_policies)):
            self.cancellation_policies_index[self.cancellation_policies[i]] = i
        self.df.cancellation_policy = self.df.cancellation_policy.apply(
            lambda x: self.cancellation_policies_index[x]).astype('category')

        # convert response time to an index
        self.response_times = self.df.host_response_time.unique().tolist()
        self.response_times_index = {}
        for i in range(len(self.response_times)):
            self.response_times_index[self.response_times[i]] = i
        self.df.host_response_time = self.df.host_response_time.apply(lambda x: self.response_times_index[x]).astype(
            'category')

        # convert binary values to bool
        self.df.host_is_superhost = self.df.host_is_superhost.apply(lambda x: x == 't')
        self.df.host_has_profile_pic = self.df.host_has_profile_pic.apply(lambda x: x == 't')
        self.df.host_identity_verified = self.df.host_identity_verified.apply(lambda x: x == 't')
        self.df.has_availability = self.df.has_availability.apply(lambda x: x == 't')
        self.df.availability_30 = self.df.availability_30.apply(lambda x: x == 't')
        self.df.availability_60 = self.df.availability_60.apply(lambda x: x == 't')
        self.df.availability_90 = self.df.availability_90.apply(lambda x: x == 't')
        self.df.availability_365 = self.df.availability_365.apply(lambda x: x == 't')
        self.df.requires_license = self.df.requires_license.apply(lambda x: x == 't')
        self.df.instant_bookable = self.df.instant_bookable.apply(lambda x: x == 't')
        self.df.is_business_travel_ready = self.df.is_business_travel_ready.apply(lambda x: x == 't')
        self.df.require_guest_profile_picture = self.df.require_guest_profile_picture.apply(lambda x: x == 't')
        self.df.require_guest_phone_verification = self.df.require_guest_phone_verification.apply(lambda x: x == 't')

        # convert dollars to floats
        self.dollarToFloat('price')
        self.dollarToFloat('extra_people')
        self.fillNulls('cleaning_fee', '0')  # Assuming missing cleaning fees indicates 0 charge.
        self.dollarToFloat('cleaning_fee')
        self.fillNulls('security_deposit', '0')
        self.dollarToFloat('security_deposit')

        # convert host response rate to float
        self.df.host_response_rate = self.df.host_response_rate.str.replace('%', '').astype(float) / 100
        self.fillNulls('host_response_rate', self.df.host_response_rate.mean())

        # clean up other null vals
        self.fillNulls('bathrooms', 1.0)
        self.fillNulls('bedrooms', 1.0)
        self.fillNulls('beds', 1.0)

        self.fillNulls('review_scores_rating', self.df.review_scores_rating.mean())
        self.fillNulls('review_scores_accuracy', self.df.review_scores_accuracy.mean())
        self.fillNulls('review_scores_cleanliness', self.df.review_scores_cleanliness.mean())
        self.fillNulls('review_scores_checkin', self.df.review_scores_checkin.mean())
        self.fillNulls('review_scores_communication', self.df.review_scores_communication.mean())
        self.fillNulls('review_scores_location', self.df.review_scores_location.mean())
        self.fillNulls('review_scores_value', self.df.review_scores_value.mean())
        self.fillNulls('reviews_per_month', 0)

        # set kmeans neighborhoods using estimated best k=8 option
        self.longLat(k=8)

        if self.groupAmenities:
            self.df.amenities = self.df.amenities.apply(lambda x: len(x.split(',')))
        else:
            amenities = set()
            for listing in self.df.amenities:
                replacements = ['{', '}', '"']
                for r in replacements:
                    listing = listing.replace(r, '').lower()
                spacers = ['/', ':', ';', '-', '(', ')', '&']
                for s in spacers:
                    listing = listing.replace(s, '_')
                l = listing.split(',')
                for am in l:
                    amenities.add(am)
            for amenity in amenities:
                if amenity != "" and 'missing' not in amenity:
                    self.df[amenity] = self.df.amenities.apply(lambda x: amenity in x)

            self.df = self.df.drop('amenities', axis=1)
            self.columns = self.df.columns.values.tolist()

        # Drop outliers past the 95% quantile
        q = self.df.price.quantile(0.95)
        self.df = self.df[self.df.price <= q]

        self.df = self.df.drop(['id', 'longitude', 'latitude'], axis=1)

        self.y = self.df.price
        self.x = self.df.drop(['price'], axis=1)
        self.y.to_csv('y.csv', index=None, header=True)
        self.x.to_csv('x.csv', index=None, header=True)

import sklearn.metrics as metrics
def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred)
    mse=metrics.mean_squared_error(y_true, y_pred)
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))


from sklearn.metrics import pairwise_distances_argmin


def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)

        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])

        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels





def test():
    #test = Listings()
    #test.readListings('/Users/joelnorthrup/Desktop/listings_summary.csv')
    #test.df.drop(['id', 'listing_url', 'scrape_id', 'last_scraped', 'name', 'summary', 'space', 'description', 'experiences_offered', 'neighborhood_overview', 'notes', 'transit', 'access', 'interaction', 'house_rules', 'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url', 'host_id', 'host_url', 'host_name', 'host_since', 'host_location', 'host_about', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost', 'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood', 'host_listings_count', 'host_total_listings_count', 'host_verifications', 'host_has_profile_pic', 'host_identity_verified', 'street', 'neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'city', 'state', 'zipcode', 'market', 'smart_location', 'country_code', 'country', 'is_location_exact', 'property_type', 'bedrooms', 'beds', 'bed_type', 'amenities', 'square_feet', 'price', 'weekly_price', 'monthly_price', 'security_deposit', 'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights', 'calendar_updated', 'has_availability', 'availability_30', 'availability_60', 'availability_90', 'availability_365', 'calendar_last_scraped', 'number_of_reviews', 'first_review', 'last_review', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_value', 'requires_license', 'license', 'jurisdiction_names', 'instant_bookable', 'is_business_travel_ready', 'cancellation_policy', 'require_guest_profile_picture', 'require_guest_phone_verification'])


    #test.dropCols(toDrop= ['id', 'listing_url', 'scrape_id', 'last_scraped', 'name', 'summary', 'space', 'description', 'experiences_offered', 'neighborhood_overview', 'notes', 'transit', 'access', 'interaction', 'house_rules', 'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url', 'host_id', 'host_url', 'host_name', 'host_since', 'host_location', 'host_about', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost', 'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood', 'host_listings_count', 'host_total_listings_count', 'host_verifications', 'host_has_profile_pic', 'host_identity_verified', 'street', 'neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'city', 'state', 'zipcode', 'market', 'smart_location', 'country_code', 'country', 'is_location_exact', 'property_type', 'bedrooms', 'beds', 'bed_type', 'amenities', 'square_feet', 'price', 'weekly_price', 'monthly_price', 'security_deposit', 'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights', 'calendar_updated', 'has_availability', 'availability_30', 'availability_60', 'availability_90', 'availability_365', 'calendar_last_scraped', 'number_of_reviews', 'first_review', 'last_review', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_value', 'requires_license', 'license', 'jurisdiction_names', 'instant_bookable', 'is_business_travel_ready', 'cancellation_policy', 'require_guest_profile_picture', 'require_guest_phone_verification'])
    #test.dr

    # test.fillNulls('bathrooms', 1.0)
    # test.fillNulls('reviews_per_month', 0)
    # testModel = Modeler(test.x , test.y)
    # print(test.x)
    # testModel.split_data()
    # testModel.principalComponentAnalysis()


    listing = Listings()
    listing.readListings('/Users/joelnorthrup/Desktop/listings_summary.csv')
    listing.longLat()
    listing.dropCols()
    listing.cleaner()
    model = Modeler(listing.x , listing.y)
    model.split_data(.5)
    model.ridger()
    model.plotModel()
    #regression_results()



    #model.principalComponentAnalysis()

    test_df = pd.read_csv('/Users/joelnorthrup/Desktop/berlin_x_test.csv', low_memory=False)
    test_df.set_index('id')
    test_df.columns = test_df.columns.values.tolist()
    import statistics

    for i in range(len(test_df['price'])):
        if test_df['price'][i] >= 1000:
            test_df['price'][i] = statistics.mean(test_df['price'])




    pca = PCA(n_components=1)  # project from 171 to 2 dimensions
    # pca.fit(self.x_train)
    projected = pca.fit_transform(test_df[['accommodates', 'room_type', 'longitude', 'bathrooms', 'cleaning_fee', 'review_scores_location', 'calculated_host_listings_count_shared_rooms', 'latitude']])

    #accommodates+room_type+longitude+bathrooms+cleaning_fee+review_scores_location+reviews_per_month+calculated_host_listings_count_shared_rooms


    # X_pca = pca.transform(self.x_train)
    print("original shape:   ", test_df.shape)
    print("transformed shape after PCA:", projected.shape)
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    # ax.plot(projected[:, 0], projected[:, 1], projected[:, 2])
    #ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2], c='r', marker='o')

    # ax.set_xlabel('Component 1')
    # ax.set_ylabel('Component 2')
    # ax.set_zlabel('Component 3')
    #
    # # plt.colorbar()
    # plt.title("PCA from 11 to 3!")
    # plt.show()

    percent_variance = np.round(pca.explained_variance_ratio_ * 100, decimals=2)
    columns = ['PC1']
    plt.bar(x=range(1, 2), height=percent_variance, tick_label=columns)
    plt.ylabel('Percentate of Variance Explained')
    plt.xlabel('Principal Component')
    plt.title('PCA Scree Plot')
    plt.show()

    # plt.scatter(projected[:, 0], test_df[['price']])
    # plt.xlabel('Component 1')
    # plt.ylabel('Price')
    # plt.title("PCA 1st Component w/ Price")
    # plt.show()

    linear_regressor = LinearRegression(normalize=True)  # create object for the class
    X = projected
    Y = test_df['price']
    linear_regressor.fit(X,Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)

    #poly = PolynomialFeatures(degree=2)
    #X_poly = poly.fit_transform(X)

    #poly.fit(X_poly, Y)
    #lin2 = LinearRegression()
    #lin2.fit(X_poly, Y)

    plt.scatter(X, Y)
    #plt.plot(X, lin2.predict(X_poly), color='black')
    plt.plot(X, Y_pred, color='red')
    plt.xlabel('Component 1')
    plt.ylabel('Price')
    plt.title("PCA 1st Component w/ Price")
    plt.show()
    #regression_results(Y,Y_pred)


    #centers, labels = find_clusters(X, 4)
    #plt.scatter(X, Y, c=labels, s=50, cmap='viridis')
    #plt.show()

#test()


# def attempt():
#     DATA = pd.read_csv('/Users/joelnorthrup/Desktop/berlinDF.csv')
#     #DATA = np.asmatrix(DATA)
#     #DATA.set_index('id')
#     #DATA.columns = DATA.columns.values.tolist()
#
#     import statistics
#
#     for i in range(len(DATA['price'])):
#         if DATA['price'][i] >= 400:
#             DATA['price'][i] = statistics.mean(DATA['price'])
#
#
#
#     pca = PCA(n_components=1, svd_solver = 'full')  # project from 171 to 2 dimensions
#     projected = pca.fit_transform(DATA[['host_is_superhost', 'host_identity_verified','host_response_rate','host_response_time', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type', 'security_deposit', 'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm']])
#     # ['host_response_time', 'host_response_rate', 'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'latitude', 'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type', 'security_deposit', 'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy',	'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'requires_license', 'instant_bookable', 'cancellation_policy', 'require_guest_profile_picture', 'require_guest_phone_verification', 'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms', 'reviews_per_month', 'kmeans_neighborhoods', 'crib', 'balcony', 'dryer', 'kitchen', 'pool', 'oven', 'washer', 'essentials', 'wifi', 'hot tub', 'toilet']
#     print("original shape:   ", DATA.shape)
#     print("transformed shape after PCA:", projected.shape)
#
#     # percent_variance = np.round(pca.explained_variance_ratio_ * 100, decimals=2)
#     # columns = ['PC1', 'PC2']
#     # plt.bar(x=range(1, 3), height=percent_variance, tick_label=columns)
#     # plt.ylabel('Percentate of Variance Explained')
#     # plt.xlabel('Principal Component')
#     # plt.title('PCA Scree Plot')
#     # plt.show()
#
#     linear_regressor = LinearRegression(normalize=True)  # create object for the class
#     X = projected
#     Y = DATA['price']
#     linear_regressor.fit(X, Y)  # perform linear regression
#     Y_pred = linear_regressor.predict(X)
#
#     # poly = PolynomialFeatures(degree=2)
#     # X_poly = poly.fit_transform(X)
#
#     # poly.fit(X_poly, Y)
#     # lin2 = LinearRegression()
#     # lin2.fit(X_poly, Y)
#
#     plt.scatter(X, Y)
#     # plt.plot(X, lin2.predict(X_poly), color='black')
#     plt.plot(X, Y_pred, color='red')
#     plt.xlabel('Component 1')
#     plt.ylabel('Price')
#     plt.title("PCA 1st Component w/ Price")
#     plt.show()
#
#     # centers, labels = find_clusters(projected, 2)
#     # plt.scatter(projected[:,0], projected[:,1], c=labels, s=50, cmap='viridis')
#
#     # plt.title('K-means Clustering with 2 dimensions')
#     # plt.show()
#
# attempt()




#def test3():

    # test_df = pd.read_csv('/Users/joelnorthrup/Desktop/berlin_x_test.csv', low_memory=False)
    # test_df.set_index('id')
    # test_df.columns = test_df.columns.values.tolist()
    # import statistics
    #
    # for i in range(len(test_df['price'])):
    #     if test_df['price'][i] >= 1000:
    #         test_df['price'][i] = statistics.mean(test_df['price'])
    #
    #
    #
    #
    # pca = PCA(n_components=1)  # project from 171 to 2 dimensions
    # # pca.fit(self.x_train)
    # projected = pca.fit_transform(test_df[['accommodates', 'room_type', 'longitude', 'bathrooms', 'cleaning_fee', 'review_scores_location', 'calculated_host_listings_count_shared_rooms', 'latitude']])




from geopy.distance import great_circle

DATA = pd.read_csv('/Users/joelnorthrup/Desktop/chicago_x.csv')
YDATA = pd.read_csv('/Users/joelnorthrup/Desktop/chicago_y.csv')

DATA['longitude']

import statistics
#
# for i in range(len(YDATA['price'])):
#     if YDATA['price'][i] >= 600:
#         YDATA['price'][i] = statistics.mean(YDATA['price'])




def distance_to_mid(lat, lon):
    chicago_center = (41.8781, -87.6298)
    mill_park = (41.8826, -87.6226)
    accommodation = (lat, lon)
    return great_circle(chicago_center, accommodation).miles

DATA['distance'] = DATA.apply(lambda x: distance_to_mid(x.latitude, x.longitude), axis=1)

DATA['distance']

linear_regressor = LinearRegression(normalize=True)  # create object for the class
X = DATA[['distance']]
Y = YDATA[['price']]
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.xlabel('Distance')
plt.ylabel('Price')
plt.title("Predicting Price From Distance to Center")
plt.show()

regression_results(abs(Y),abs(Y_pred))

pca = PCA(n_components=1)  # project from 171 to 1 dimensions
projected = pca.fit_transform(DATA[['distance', 'latitude', 'longitude', 'room_type', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count','availability_365']])
# ['host_response_time', 'host_response_rate', 'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'latitude', 'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type', 'security_deposit', 'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy',	'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'requires_license', 'instant_bookable', 'cancellation_policy', 'require_guest_profile_picture', 'require_guest_phone_verification', 'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms', 'reviews_per_month', 'kmeans_neighborhoods', 'crib', 'balcony', 'dryer', 'kitchen', 'pool', 'oven', 'washer', 'essentials', 'wifi', 'hot tub', 'toilet']
print("original shape:   ", DATA.shape)
print("transformed shape after PCA:", projected.shape)

# percent_variance = np.round(pca.explained_variance_ratio_ * 100, decimals=2)
# columns = ['PC1', 'PC2']
# plt.bar(x=range(1, 3), height=percent_variance, tick_label=columns)
# plt.ylabel('Percentate of Variance Explained')
# plt.xlabel('Principal Component')
# plt.title('PCA Scree Plot')
# plt.show()

linear_regressor = LinearRegression(normalize=True)  # create object for the class
X = projected
linear_regressor.fit(X, YDATA)  # perform linear regression
Y_pred = linear_regressor.predict(X)
plt.scatter(X, YDATA)
plt.plot(X, Y_pred, color='red')
plt.xlabel('PC1')
plt.ylabel('Price')
plt.title("Predicting Price From PCA")
plt.show()
regression_results(abs(Y),abs(Y_pred))






############################################################################################
############################################################################################

import esda
import rtree
import pandas as pd
import geopandas as gpd
# from geopandas import GeoDataFrame
# import libpysal as lps
# import numpy as np
# import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, MultiPolygon
# #import libpysal.api as lp
import matplotlib.pyplot as plt
#import rasterio as rio
import numpy as np
#import contextily as ctx
import shapely.geometry as geom

#gdf = gpd.read_file('/Users/joelnorthrup/Desktop/neighbourhoods.geojson').to_crs(epsg=3857)
#bl_df = pd.read_csv('/Users/joelnorthrup/Desktop/listings.csv')

# geometry = [Point(xy) for xy in zip(bl_df.longitude, bl_df.latitude)]
# crs = {'init': 'epsg:4326'}
# bl_gdf = GeoDataFrame(bl_df, crs=crs, geometry=geometry)

#listings['geometry'] = listings[['longitude','latitude']].apply(geom.Point, axis=1) #take each row



# bl_gdf['price'] = bl_gdf['price'].astype('float32')
#
# # listings = gpd.GeoDataFrame(listings)
# # listings.crs={'init':'epsg:4269'}
# # listings = listings.to_crs(epsg=3857)
# #
# #
# # basemap, bounds = ctx.bounds2img(*listings.total_bounds, zoom=10, url='http://tile.stamen.com/toner-lite/tileZ/tileX/tileY.png')
#
# gdf.reset_index(drop=True)
# bl_gdf.reset_index(drop=True)
# sj_gdf = gpd.sjoin(gdf, bl_gdf,how='inner')
# # how="inner", op="intersects", lsuffix="left", rsuffix="right"
# median_price_gb = sj_gdf['price'].groupby([sj_gdf['neighbourhood_group']]).mean()
# median_price_gb










# gdf = gpd.read_file('/Users/joelnorthrup/Desktop/neighbourhoods.geojson').to_crs(epsg=3857)
# bl_df = pd.read_csv('/Users/joelnorthrup/Desktop/listings.csv')


# 1-  Listing points
# listings = pd.read_csv('/Users/joelnorthrup/Desktop/listings.csv')
# # 2 - convert to Geopandas Geodataframe
# gdf_listings = gpd.GeoDataFrame(listings,   geometry=gpd.points_from_xy(listings.longitude, listings.latitude))
# # 3 - Neighbourhoods
# geojson_file = '/Users/joelnorthrup/Desktop/neighbourhoods.geojson'
# neighborhoods = gpd.read_file(geojson_file)
#
# gdf_listings.head(2)
#
# "/Users/joelnorthrup/Desktop/Boundaries - Community Areas (current).geojson"
#
#
# sjoined_listings = gpd.sjoin(gdf_listings, neighborhoods, op='within')
# sjoined_listings.head()

listings = pd.read_csv("/Users/joelnorthrup/Desktop/listings.csv")
gdf_listings = gpd.GeoDataFrame(listings,crs={'init' :'epsg:4326'}, geometry=gpd.points_from_xy(listings.longitude, listings.latitude))
neighborhoods = gpd.read_file("/Users/joelnorthrup/Desktop/neighbourhoods.geojson")
gdf_listings.head()

gdf_listings[["id", "name","host_name", "neighbourhood", "latitude","longitude","room_type","price","reviews_per_month", "geometry"]].head()


gdf_listings["geometry"].isnull().sum()

gdf_listings['number_of_reviews']


neighborhoods.crs, gdf_listings.crs

neighborhoods.head()

fig, ax = plt.subplots(figsize=(12,10))
neighborhoods.plot(color="Gray", ax= ax)
gdf_listings.plot(ax=ax)
plt.show()
plt.savefig("neighbourhoods.png")


chicago_areas = gpd.read_file("/Users/joelnorthrup/Desktop/chicago-community-areas.geojson")
chicago_areas.head()

fig, ax = plt.subplots(figsize=(12,10))
chicago_areas.plot(color="Gray", ax= ax)
gdf_listings.plot(ax=ax, markersize=1)
plt.show()



merged_listings = chicago_areas.merge(gdf_listings, on='geometry')
merged_listings


chicago_areas[['geometry']][0:1].area
#sjoined_listings = gpd.sjoin(gdf_listings[['geometry']][0:1], chicago_areas[['geometry']][0:1], op="contains")

chicago_areas[['geometry']][:4].contains(gdf_listings[['geometry']][0:])

# Reproject to Albers for plotting
country_albers = chicago_areas.to_crs({'init': 'epsg:5070'})
roads_albers = gdf_listings.to_crs({'init': 'epsg:5070'})

roads_albers['distance'] = roads_albers.apply(lambda x: distance_to_mid(x.latitude, x.longitude), axis=1)


for i in range(len(roads_albers['price'])):
    if roads_albers['price'][i] != 0:
        roads_albers['price'][i] = np.math.log(roads_albers['price'][i])

roads_albers['price']




linear_regressor = LinearRegression(normalize=True)  # create object for the class
X = roads_albers[['distance']]

linear_regressor.fit(X, roads_albers[['price']])  # perform linear regression
Y_pred = linear_regressor.predict(X)
plt.scatter(X, roads_albers[['price']])
plt.plot(X, Y_pred, color='red')
plt.xlabel('Distance')
plt.ylabel('Price')
plt.title("TEST")
plt.show()
regression_results(abs(roads_albers[['price']]),abs(Y_pred))




# Plot the data
fig, ax = plt.subplots(figsize=(12, 8))
country_albers.plot(alpha=1,facecolor="none",edgecolor="black",zorder=10,ax=ax)
roads_albers.plot(alpha=1, ax=ax, column ='price', legend=True)
# Adjust legend location
#leg = ax.get_legend()
#leg.set_bbox_to_anchor((1.15,1))

ax.set_axis_off()
plt.axis('equal')
plt.title('CITY OF CHICAGO \n color-coded by LOG Price')
plt.show()

plt.hist(roads_albers['price'], color = 'blue', edgecolor = 'black', bins = int(180/5))

sns.distplot(roads_albers['price'], hist=True, kde=False, bins=int(180/5), color = 'blue',hist_kws={'edgecolor':'black'})
# Add labels
plt.title('Histogram of Price')
plt.xlabel('Price')
plt.ylabel('count')
plt.show()

roads_albers['price']

aves = roads_albers.groupby(roads_albers['room_type'])['price'].mean()

aves.plot.bar(figsize=(10,4))
plt.title('Average Price Per Room Type')
plt.xlabel('Room Type')
plt.ylabel('Price')
plt.show()



#******************************************************************************#




aves.plot.bar(figsize=(10,4))
plt.title('Average Price Per Neighborhood')
plt.show()


sns.violinplot(x=roads_albers['neighbourhood'], y=roads_albers['price'])
plt.show()


plt.hist(aves, color = 'blue', edgecolor = 'black', bins = int(180/5))

sns.distplot(aves, hist=True, kde=False, bins=int(180/5), color = 'blue',hist_kws={'edgecolor':'black'})
# Add labels
plt.title('Histogram of Price')
plt.xlabel('Price')
plt.ylabel('count')
plt.show()


fig, ax = plt.subplots(figsize=(12, 8))
country_albers.plot(alpha=1,facecolor="none",edgecolor="black",zorder=10,ax=ax)
roads_albers.plot(ax=ax,column = 'distance', legend=True)
# Adjust legend location
#leg = ax.get_legend()
#leg.set_bbox_to_anchor((1.15,1))

ax.set_axis_off()
plt.axis('equal')
plt.title('color-coded by Distance')
plt.show()


##################

abb_link = '/Users/joelnorthrup/Desktop/listings.csv'
zc_link = "/Users/joelnorthrup/Desktop/chicago-community-areas.geojson"

lst = pd.read_csv(abb_link)
lst
varis = ['price']

lst['price']
aves = lst.groupby(lst['neighbourhood'])[varis].mean()
aves.info()

types = pd.get_dummies(lst['room_type'])
prop_types = types.join(lst['neighbourhood']).groupby('neighbourhood').sum()
prop_types_pct = (prop_types * 100.).div(prop_types.sum(axis=1), axis=0)
prop_types_pct.info()

aves_props = aves.join(prop_types_pct)

db = pd.DataFrame(scale(aves_props), index=aves_props.index, columns=aves_props.columns).rename(lambda x: str(x))

zc = gpd.read_file(zc_link)



zc.plot(color='red')
plt.show()

zc.head(2)

zdb = zc.join(db).dropna()

f, ax = plt.subplots(1, figsize=(9, 9))

zc.plot(color='grey', linewidth=0, ax=ax)
zdb.plot(color='red', linewidth=0.1, ax=ax)

zdb

ax.set_axis_off()

plt.show()


#km5 = KMeans(n_clusters=5)

#km5cls = km5.fit(zdb)
