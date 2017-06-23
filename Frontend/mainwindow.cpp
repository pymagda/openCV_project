#include <qfiledialog.h>
#include <qmessagebox.h>
#include <sstream>
#include <thread>
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "Backend/Include/CmCut.h"


MainWindow::MainWindow(QWidget *parent) :
        QMainWindow(parent),
        ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_browse_images_clicked()
{
    QString path;

    path = QFileDialog::getOpenFileName(
            this,
            "Choose an image to process",
            QString::null,
            QString::null);

    ui->lineEdit_image_path->setText(path);
}

void MainWindow::on_pushButton_process_image_clicked()
{

    if(ui->lineEdit_image_path->text().isEmpty())
    {
        QMessageBox::about(this,"Error","Choose file to process!");
        return;
    }
    std::string file_path = ui->lineEdit_image_path->text().toUtf8().constData();
    if(!(file_path.substr(file_path.length()-4,4) == ".jpg" || file_path.substr(file_path.length()-5,5) == ".JPEG"
     || file_path.substr(file_path.length()-4,4) == ".png" || file_path.substr(file_path.length()-4,4) == ".JPG"
    ))
    {
        QMessageBox::about(this,"Error","Only JPG or PNG files are supported!");
        return;
    }

    std::string image_path = ui->lineEdit_image_path->text().toUtf8().constData();
    ui->lineEdit_image_path->clear();

    std::string image_result_path;
    std::thread image_process_thread(CmCut::Demo, std::ref(image_path), std::ref(image_result_path));
    image_process_thread.join();

    QPixmap pix(image_result_path.c_str());
    ui->label_result_image->setPixmap(pix.scaled(400,400,Qt::KeepAspectRatio));
}
