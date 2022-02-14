from ilqr_solver import LogBarrieriLQR, NNiLQR
from scenario.car_parking import CarParking
from scenario.quadcopter import QuadCopter
from scenario.robotic_arm_tracking import RoboticArmTracking
from utils.Logger import logger
import matplotlib.pyplot as plt

# if __name__ == "__main__":
#     for i in range(10):
#         try:
#             logger.set_folder_name("QuadCopter_" + str(i), remove_existing_folder=False).set_is_use_logger(True).set_is_save_json(True)
#             scenario = QuadCopter() 
#             NNiLQR(gaussian_noise_sigma=[[0.1], [0.1], [0.1], [0.1]], iLQR_max_iter=100).init(scenario).solve() 
#         except Exception as e:
#             pass
#         continue
#     scenario = QuadCopter() 
#     scenario.play("QuadCopter_9")

#     logger.set_folder_name("QuadCopter_Log", remove_existing_folder=False).set_is_use_logger(True).set_is_save_json(True)
#     scenario = QuadCopter() 
#     LogBarrieriLQR().init(scenario).solve() 
#     scenario.play("QuadCopter_Log")


# if __name__ == "__main__":
#     for i in range(10):
#         try:
#             logger.set_folder_name("CarParking_" + str(i), remove_existing_folder=False).set_is_use_logger(True).set_is_save_json(True)
#             scenario = CarParking() 
#             NNiLQR(gaussian_noise_sigma=[[0.01], [0.1]], iLQR_max_iter=100).init(scenario).solve() 
#         except Exception as e:
#             pass
#         continue
#     scenario = CarParking() 
#     scenario.play("CarParking_9")

#     logger.set_folder_name("CarParking_Log", remove_existing_folder=False).set_is_use_logger(True).set_is_save_json(True)
#     scenario = CarParking() 
#     LogBarrieriLQR().init(scenario).solve() 
#     scenario.play("CarParking_Log")

if __name__ == "__main__":
    for i in range(10):
        try:
            logger.set_folder_name("RoboticArmTracking_" + str(i), remove_existing_folder=False).set_is_use_logger(True).set_is_save_json(True)
            scenario = RoboticArmTracking() 
            NNiLQR(gaussian_noise_sigma=[[0.1], [0.1]], iLQR_max_iter=100, training_stopping_criterion=0.01, decay_rate_max_iters=200).init(scenario).solve() 
        except Exception as e:
            pass
        continue
    scenario = RoboticArmTracking() 
    scenario.play("RoboticArmTracking_9")

    logger.set_folder_name("RoboticArmTracking_log", remove_existing_folder=False).set_is_use_logger(True).set_is_save_json(True)
    scenario = RoboticArmTracking() 
    LogBarrieriLQR().init(scenario).solve() 
    scenario.play("RoboticArmTracking_log")
    plt.show()

