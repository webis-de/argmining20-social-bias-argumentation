#! /bin/bash

python prepare_iac_data.py \
    --mysql_address "localhost" \
    --mysql_user "mysql" \
    --mysql_password "mysql" \
    --convinceme_db_name "convinceme" \
    --createdebate_db_name "createdebate" \
    --fourforums_db_name "fourforums" \
    --output "output"
