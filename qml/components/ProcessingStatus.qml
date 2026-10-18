pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../theme" as Room36Theme

Rectangle {
    id: root
    objectName: "processingStatus"
    property var controller: null
    Room36Theme.Theme { id: theme }
    height: controller ? 100 : 0
    visible: controller !== null
    color: theme.panelFill
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 8

        RowLayout {
            Layout.fillWidth: true
            Text {
                text: !root.controller ? "" : root.controller.overallState === "COMPLETE" ? "Project ready"
                    : root.controller.overallState === "FAILED" ? "Some analysis needs attention"
                    : root.controller.overallState === "NOT_STARTED" ? "Waiting for project"
                    : "Processing photos…"
                color: theme.mainBlue
                Accessible.name: text
            }
            Item { Layout.fillWidth: true }
            PrimaryButton {
                objectName: "processingRetry"
                text: "Retry"
                visible: root.controller !== null && root.controller.overallState === "FAILED"
                onClicked: root.controller.retry()
                Accessible.name: "Retry project processing"
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            Repeater {
                model: root.controller ? root.controller.stages : []
                delegate: ColumnLayout {
                    id: stageItem
                    required property var modelData
                    readonly property string stageState: modelData.state
                    readonly property string statusText: stageState === "COMPLETE" ? "Done"
                        : stageState === "RUNNING" && modelData.total > 0
                            ? modelData.done + "/" + modelData.total
                        : stageState === "RUNNING" ? "Working"
                        : stageState === "FAILED" ? "Failed"
                        : stageState === "CANCELLED" ? "Canceled"
                        : stageState === "QUEUED" ? "Queued" : "Waiting"
                    Layout.fillWidth: true
                    Layout.minimumWidth: 0
                    spacing: 4

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 4
                        Text {
                            Layout.fillWidth: true
                            text: stageItem.modelData.label
                            elide: Text.ElideRight
                            color: theme.mainBlue
                            font.pixelSize: 11
                        }
                        Text {
                            text: stageItem.statusText
                            color: theme.mutedText
                            font.pixelSize: 10
                        }
                    }
                    ProgressBar {
                        id: stageProgress
                        objectName: "processingProgress_" + stageItem.modelData.stage
                        Layout.fillWidth: true
                        from: 0
                        to: 1
                        value: stageItem.stageState === "COMPLETE" ? 1
                            : stageItem.modelData.total > 0
                                ? Math.max(0, Math.min(1, stageItem.modelData.done / stageItem.modelData.total)) : 0
                        indeterminate: stageItem.stageState === "RUNNING" && stageItem.modelData.total <= 0
                        Accessible.name: stageItem.modelData.label + ": " + stageItem.statusText

                        background: Rectangle {
                            implicitHeight: 8
                            radius: height / 2
                            color: theme.separator
                        }
                        contentItem: Item {
                            implicitHeight: 8
                            clip: true
                            Rectangle {
                                id: progressFill
                                objectName: "processingProgressFill"
                                height: parent.height
                                width: stageProgress.indeterminate
                                    ? parent.width * 0.35 : parent.width * stageProgress.visualPosition
                                x: stageProgress.indeterminate ? sweep : 0
                                visible: stageProgress.indeterminate || stageProgress.value > 0
                                radius: height / 2
                                property real sweep: -width
                                gradient: Gradient {
                                    GradientStop { position: 0; color: theme.lightBlue }
                                    GradientStop { position: 0.5; color: theme.midBlue }
                                    GradientStop { position: 1; color: theme.mainBlue }
                                }
                                NumberAnimation on sweep {
                                    from: -progressFill.width
                                    to: progressFill.parent.width
                                    duration: 1200
                                    loops: Animation.Infinite
                                    running: stageProgress.indeterminate
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
