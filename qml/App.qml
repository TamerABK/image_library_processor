import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "theme" as Room36Theme

ApplicationWindow {
    visible: true
    width: 1440
    height: 900
    minimumWidth: metrics.minimumWindowWidth
    minimumHeight: metrics.minimumWindowHeight
    title: "Room 36 / Qt migration"
    color: theme.white

    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    Room36Theme.Metrics { id: metrics }

    header: Rectangle {
        color: theme.mainBlue
        implicitHeight: 92

        ColumnLayout {
            anchors.fill: parent
            anchors.leftMargin: metrics.spacingLarge
            anchors.rightMargin: metrics.spacingLarge
            anchors.topMargin: metrics.spacingMedium
            anchors.bottomMargin: metrics.spacingMedium
            spacing: metrics.spacingTiny

            Text {
                text: "Room 36 / Qt migration"
                color: theme.white
                font.family: typography.boldFamily
                font.pixelSize: typography.heroSize
            }

            Text {
                text: "Phase 1 development shell"
                color: theme.lightBlue
                font.family: typography.regularFamily
                font.pixelSize: typography.bodySize
            }
        }
    }

    Rectangle {
        anchors.fill: parent
        color: theme.white

        ColumnLayout {
            anchors.fill: parent
            anchors.margins: metrics.spacingLarge
            spacing: metrics.spacingLarge

            Rectangle {
                Layout.fillWidth: true
                radius: metrics.radiusMedium
                border.width: 1
                border.color: theme.lightBlue
                color: theme.white
                implicitHeight: shellLayout.implicitHeight + (metrics.spacingLarge * 2)

                ColumnLayout {
                    id: shellLayout
                    anchors.fill: parent
                    anchors.margins: metrics.spacingLarge
                    spacing: metrics.spacingMedium

                    Text {
                        Layout.fillWidth: true
                        text: "Temporary bootstrap surface. Final Room 36 screens are intentionally not implemented in this phase."
                        wrapMode: Text.Wrap
                        color: theme.black
                        font.family: typography.regularFamily
                        font.pixelSize: typography.bodySize
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: metrics.spacingSmall

                        TextField {
                            id: folderInput
                            Layout.fillWidth: true
                            placeholderText: "Photo library folder"
                            text: appBridge.folder
                            font.family: typography.regularFamily
                            font.pixelSize: typography.bodySize
                            selectByMouse: true
                            onEditingFinished: {
                                appBridge.setFolder(text)
                                appBridge.refreshFileTypes()
                            }
                        }

                        Button {
                            text: "Apply Folder"
                            onClicked: {
                                appBridge.setFolder(folderInput.text)
                                appBridge.refreshFileTypes()
                            }
                        }

                        Button {
                            text: "Browse"
                            onClicked: folderInput.text = appBridge.chooseFolder()
                        }
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: metrics.spacingTiny

                        Text {
                            Layout.fillWidth: true
                            text: "Folder: " + (appBridge.folder.length > 0 ? appBridge.folder : "Not set")
                            wrapMode: Text.WrapAnywhere
                            color: theme.black
                            font.family: typography.regularFamily
                            font.pixelSize: typography.bodySize
                        }

                        Text {
                            Layout.fillWidth: true
                            text: "Status: " + appBridge.status
                            wrapMode: Text.Wrap
                            color: theme.black
                            font.family: typography.regularFamily
                            font.pixelSize: typography.bodySize
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: metrics.spacingSmall

                            Text {
                                text: appBridge.countText
                                color: theme.midBlue
                                font.family: typography.regularFamily
                                font.pixelSize: typography.bodySize
                            }

                            Rectangle {
                                width: 6
                                height: 6
                                radius: 3
                                color: theme.neonGreen
                            }

                            Text {
                                text: appBridge.elapsedText
                                color: theme.black
                                font.family: typography.regularFamily
                                font.pixelSize: typography.bodySize
                            }

                            Item { Layout.fillWidth: true }

                            Text {
                                text: typography.hasBrandFonts ? "Britanica loaded" : "Fallback fonts in use"
                                color: typography.hasBrandFonts ? theme.midBlue : theme.black
                                font.family: typography.regularFamily
                                font.pixelSize: typography.labelSize
                            }
                        }
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: metrics.spacingTiny

                        ProgressBar {
                            Layout.fillWidth: true
                            from: 0
                            to: Math.max(1, appBridge.progressMaximum)
                            value: appBridge.progressValue
                            indeterminate: appBridge.progressIndeterminate
                        }

                        Text {
                            text: appBridge.progressIndeterminate
                                ? "Progress: working"
                                : "Progress: " + appBridge.progressValue + " / " + appBridge.progressMaximum
                            color: theme.black
                            font.family: typography.regularFamily
                            font.pixelSize: typography.labelSize
                        }
                    }

                    RowLayout {
                        spacing: metrics.spacingSmall

                        Button {
                            text: "Start Scan"
                            enabled: appBridge.canScan
                            onClicked: appBridge.startScan()
                        }

                        Button {
                            text: "Cancel"
                            enabled: appBridge.canCancel
                            onClicked: appBridge.cancelScan()
                        }
                    }
                }
            }

            Item { Layout.fillHeight: true }
        }
    }

    Connections {
        target: appBridge

        function onFolderChanged() {
            if (!folderInput.activeFocus) {
                folderInput.text = appBridge.folder
            }
        }
    }
}
