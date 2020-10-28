#! /bin/bash

MYSQL_USER="mysql"
MYSQL_PASSWORD="mysql"

CONVINCEME_DUMP_PATH="./data/iac/convinceme_2016_05_18.sql"
CREATEDEBATE_DUMP_PATH="./data/iac/createdebate_released_2016_05_18.sql"
FOURFORUMS_DUMP_PATH="./data/iac/fourforums_2016_05_18.sql"

echo ""
echo "This script assumes a running MySQL/MariaDB setup."
echo "Please also make sure to set the MySQL variables and data paths on the top of this script correctly."
echo "Currently, values are set to the following:"
echo "    MySQL user: ${MYSQL_USER}"
echo "    MySQL password: ${MYSQL_PASSWORD}"
echo "    Convinceme dump: ${CONVINCEME_DUMP_PATH}"
echo "    CreateDebate dump: ${CREATEDEBATE_DUMP_PATH}"
echo "    4Forums dump: ${FOURFORUMS_DUMP_PATH}"
echo "================================================================================"

echo ""
echo "Working on ConvinceMe portal..."
echo "Creating database."
mysql \
    --user=${MYSQL_USER} \
    --password=${MYSQL_PASSWORD} \
    -e "CREATE SCHEMA convinceme DEFAULT CHARACTER SET utf8mb4 DEFAULT COLLATE utf8mb4_bin; SET GLOBAL foreign_key_checks=0"

echo "Importing dump."
mysql \
    --user=${MYSQL_USER} \
    --password=${MYSQL_PASSWORD} \
    convinceme < ${CONVINCEME_DUMP_PATH}
echo "Done."

echo ""
echo "Working on CreateDebate portal..."
echo "Creating database."
mysql \
    --user=${MYSQL_USER} \
    --password=${MYSQL_PASSWORD} \
    -e "CREATE SCHEMA createdebate DEFAULT CHARACTER SET utf8mb4 DEFAULT COLLATE utf8mb4_bin; SET GLOBAL foreign_key_checks=0"

echo "Importing dump."
mysql \
    --user=${MYSQL_USER} \
    --password=${MYSQL_PASSWORD} \
    createdebate < ${CREATEDEBATE_DUMP_PATH}
echo "Done."

echo ""
echo "Working on 4Forums portal..."
echo "Creating database."
mysql \
    --user=${MYSQL_USER} \
    --password=${MYSQL_PASSWORD} \
    -e "CREATE SCHEMA fourforums DEFAULT CHARACTER SET utf8mb4 DEFAULT COLLATE utf8mb4_bin; SET GLOBAL foreign_key_checks=0"

echo "Importing dump."
mysql \
    --user=${MYSQL_USER} \
    --password=${MYSQL_PASSWORD} \
    fourforums < ${FOURFORUMS_DUMP_PATH}
echo "Done."
